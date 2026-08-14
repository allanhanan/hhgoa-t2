"""Audio stream handling: microphone input with VAD, speaker output."""

from __future__ import annotations

import asyncio
import struct
from typing import AsyncIterator

from voice_optimized_rag.utils.logging import get_logger

logger = get_logger("audio_stream")

# VAD frame duration in ms (must be 10, 20, or 30 for webrtcvad)
VAD_FRAME_MS = 30


class AudioStream:
    """Handles microphone input with Voice Activity Detection and speaker output.

    Uses sounddevice for audio I/O and webrtcvad for speech detection.
    Yields complete utterances (speech segments) as byte buffers.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        vad_aggressiveness: int = 2,
        silence_threshold_ms: int = 800,
        energy_threshold: float | int | None = None,
        calibration_frames: int = 20,
        energy_callback: any = None,
    ) -> None:
        self._sample_rate = sample_rate
        self._vad_aggressiveness = vad_aggressiveness
        self._silence_threshold_ms = silence_threshold_ms
        self._energy_threshold = energy_threshold
        self._calibration_frames = calibration_frames
        self._energy_callback = energy_callback
        self._frame_size = int(sample_rate * VAD_FRAME_MS / 1000)  # samples per frame

    def calculate_energy(self, frame: bytes) -> float:
        """Calculate average absolute sample amplitude (energy) of a 16-bit PCM frame."""
        if not frame:
            return 0.0
        num_samples = len(frame) // 2
        if num_samples == 0:
            return 0.0
        samples = struct.unpack(f"<{num_samples}h", frame)
        return sum(abs(sample) for sample in samples) / num_samples

    def get_effective_threshold(self, noise_floor: float) -> float:
        """Determine speech detection energy threshold based on config or ambient noise."""
        if self._energy_threshold is not None:
            return float(self._energy_threshold)
        # Default dynamic threshold calibrated to ambient noise floor:
        # Require energy to be at least 3x noise floor, or noise floor + 10, with minimum floor of 15.0
        return max(15.0, noise_floor * 3.0, noise_floor + 10.0)

    async def listen(self, yield_partials: bool = False) -> AsyncIterator[bytes | tuple[bool, bytes]]:
        """Listen to the microphone and yield complete utterances.

        If yield_partials is True, yields tuples of (is_final, bytes).
        Each yielded bytes object is a complete speech segment (from speech
        onset to silence), as 16-bit PCM audio.
        """
        try:
            import sounddevice as sd
        except ImportError:
            raise ImportError("Install sounddevice: pip install sounddevice")

        try:
            import webrtcvad
        except ImportError:
            webrtcvad = None

        try:
            input_device = sd.query_devices(kind="input")
            device_name = input_device.get("name", "Unknown Device")
            logger.info(f"Selected input device: {device_name}")
            print(f"[AudioStream] Selected input device: {device_name}")
        except Exception as err:
            logger.warning(f"Could not query input device: {err}")
            print(f"[AudioStream] Selected input device: Default (query warning: {err})")

        vad = webrtcvad.Vad(self._vad_aggressiveness) if webrtcvad else None
        audio_queue: asyncio.Queue[bytes] = asyncio.Queue()

        def audio_callback(indata, frames, time_info, status):
            if status:
                logger.warning(f"Audio status: {status}")
            audio_queue.put_nowait(bytes(indata))

        # Frame size in bytes (16-bit mono)
        frame_bytes = self._frame_size * 2
        frames_for_silence = int(self._silence_threshold_ms / VAD_FRAME_MS)

        stream = sd.RawInputStream(
            samplerate=self._sample_rate,
            blocksize=self._frame_size,
            dtype="int16",
            channels=1,
            callback=audio_callback,
        )

        try:
            with stream:
                logger.info("Audio stream started")
                print(f"[AudioStream] Stream started (sample_rate={self._sample_rate}Hz, frame_size={self._frame_size} samples)")
                buffer = bytearray()
                silent_frames = 0
                is_speaking = False
                frame_count = 0

                # Ambient noise floor calibration
                calibration_energies: list[float] = []
                noise_floor = 5.0
                effective_threshold = self.get_effective_threshold(noise_floor)

                print(f"[AudioStream] Calibrating ambient noise ({self._calibration_frames} frames)... Speak after calibration.")

                while True:
                    try:
                        frame = await audio_queue.get()
                    except (asyncio.CancelledError, KeyboardInterrupt):
                        print("[AudioStream] Stream stopped.")
                        break

                    # Ensure frame is the right size
                    if len(frame) != frame_bytes:
                        continue

                    frame_count += 1
                    energy = self.calculate_energy(frame)

                    # Initial calibration phase
                    if frame_count <= self._calibration_frames:
                        calibration_energies.append(energy)
                        if frame_count == self._calibration_frames:
                            noise_floor = sum(calibration_energies) / max(len(calibration_energies), 1)
                            effective_threshold = self.get_effective_threshold(noise_floor)
                            print(f"[AudioStream] Calibration complete! Noise floor: {noise_floor:.2f} | Dynamic threshold: {effective_threshold:.2f}")
                            print("[AudioStream] Listening for speech...")
                        continue

                    # Periodic status / audio frame received output (approx every 33 frames ~ 1 sec)
                    if frame_count % 33 == 0:
                        status_str = "SPEAKING" if is_speaking else "LISTENING"
                        print(f"[AudioStream] Audio frames received: {frame_count} | Status: {status_str} | Energy: {energy:.2f} (Threshold: {effective_threshold:.2f})")

                    # Speech detection evaluation:
                    # Check energy relative to dynamic threshold, or webrtcvad if available
                    energy_speech = energy >= effective_threshold
                    vad_speech = vad.is_speech(frame, self._sample_rate) if vad else False
                    if self._energy_callback:
                        try:
                            self._energy_callback(energy, effective_threshold, is_speech)
                        except Exception:
                            pass

                    if is_speech:
                        if not is_speaking:
                            is_speaking = True
                            logger.info(f"Speech detected! (Energy: {energy:.2f}, Threshold: {effective_threshold:.2f})")
                            print(f"[AudioStream] Speech detected! (Energy: {energy:.2f} >= {effective_threshold:.2f})")
                        buffer.extend(frame)
                        silent_frames = 0
                        
                        # Emit partials every ~0.5 seconds (approx 16 frames of 30ms)
                        if yield_partials and len(buffer) % (frame_bytes * 16) == 0:
                            yield (False, bytes(buffer))
                            
                    elif is_speaking:
                        buffer.extend(frame)
                        silent_frames += 1
                        if silent_frames >= frames_for_silence:
                            # End of utterance
                            is_speaking = False
                            silent_frames = 0
                            utterance = bytes(buffer)
                            buffer = bytearray()
                            duration_s = len(utterance) / (self._sample_rate * 2)
                            logger.info(f"Speech ended. Utterance captured: {len(utterance)} bytes ({duration_s:.2f}s)")
                            print(f"[AudioStream] Speech ended.")
                            print(f"[AudioStream] Utterance captured: {len(utterance)} bytes (~{duration_s:.2f}s)")
                            if yield_partials:
                                yield (True, utterance)
                            else:
                                yield utterance

                    # Update background noise floor slowly when silent
                    if not is_speaking and not energy_speech:
                        noise_floor = noise_floor * 0.98 + energy * 0.02
                        if self._energy_threshold is None:
                            effective_threshold = self.get_effective_threshold(noise_floor)

        except (KeyboardInterrupt, asyncio.CancelledError):
            print("\n[AudioStream] Closed cleanly.")
            return

    async def play(self, audio_data: bytes, sample_rate: int | None = None) -> None:
        """Play audio data through the speaker.

        Args:
            audio_data: Raw audio bytes. For PCM: 16-bit mono.
                       For MP3/other formats from TTS: will attempt playback directly.
            sample_rate: Override sample rate (default: self._sample_rate).
        """
        try:
            import sounddevice as sd
            import numpy as np
        except ImportError:
            raise ImportError("Install sounddevice: pip install voice-optimized-rag[voice]")

        rate = sample_rate or self._sample_rate

        # Try to interpret as 16-bit PCM
        try:
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        except ValueError:
            logger.warning("Could not interpret audio as PCM, skipping playback")
            return

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: sd.play(audio_np, rate))
        await loop.run_in_executor(None, sd.wait)
