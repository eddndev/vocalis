import init, { predict_vowel, init_vocalis } from '../pkg/vocalis_core';

const statusElement = document.getElementById('status');
const vowelElement = document.getElementById('vowel');
const genderElement = document.getElementById('gender');
const startRecordingButton = document.getElementById('startRecording');
const stopRecordingButton = document.getElementById('stopRecording');
const audioPlayback = document.getElementById('audioPlayback');
const downloadLink = document.getElementById('downloadLink');
const canvas = document.getElementById('visualizer');
const canvasCtx = canvas.getContext('2d');

const EXPECTED_SAMPLE_RATE = 16000;
const RECORDING_DURATION_MS = 1500; // 1.5 seconds for better stability
const EXPECTED_AUDIO_LENGTH = EXPECTED_SAMPLE_RATE * (RECORDING_DURATION_MS / 1000); // 8000 samples

let mediaRecorder;
let audioChunks = [];
let audioContext;
let analyser;
let visualizerFrameId;

async function loadModel() {
    statusElement.textContent = 'Status: Loading WASM module...';
    try {
        await init(); // Initialize the WASM module
        init_vocalis(); // Initialize the Rust model singleton
        statusElement.textContent = 'Status: WASM module loaded. Ready to record.';
        console.log('Vocalis WASM module loaded.');
        startRecordingButton.disabled = false;
    } catch (e) {
        statusElement.textContent = `Status: Error loading WASM module: ${e.message}`;
        console.error('Error loading WASM module:', e);
    }
}

function drawVisualizer() {
    if (!analyser) return;

    visualizerFrameId = requestAnimationFrame(drawVisualizer);

    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    analyser.getByteFrequencyData(dataArray);

    canvasCtx.fillStyle = '#1a1a1a';
    canvasCtx.fillRect(0, 0, canvas.width, canvas.height);

    const barWidth = (canvas.width / bufferLength) * 2.5;
    let barHeight;
    let x = 0;

    for (let i = 0; i < bufferLength; i++) {
        barHeight = dataArray[i];

        // Color gradient based on height (frequency magnitude)
        const r = barHeight + (25 * (i / bufferLength));
        const g = 250 * (i / bufferLength);
        const b = 50;

        canvasCtx.fillStyle = `rgb(${r},${g},${b})`;
        canvasCtx.fillRect(x, canvas.height - barHeight / 1.5, barWidth, barHeight / 1.5);

        x += barWidth + 1;
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
        console.log('Recording stopped manually');
    }
}

async function startRecording() {
    statusElement.textContent = 'Status: Requesting microphone access...';
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        console.log('Microphone access granted');
        audioContext = new (window.AudioContext || window.webkitAudioContext)();

        // Setup Visualizer
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 2048;
        const source = audioContext.createMediaStreamSource(stream);
        source.connect(analyser);
        drawVisualizer();

        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];

        mediaRecorder.ondataavailable = event => {
            audioChunks.push(event.data);
        };

        mediaRecorder.onstop = async () => {
            console.log('Recorder stopped. Processing audio...');
            cancelAnimationFrame(visualizerFrameId); // Stop visualizer

            statusElement.textContent = 'Status: Recording stopped. Processing audio...';
            const audioBlob = new Blob(audioChunks, { 'type': 'audio/webm; codecs=opus' });

            try {
                // Convert Blob to AudioBuffer
                const arrayBuffer = await audioBlob.arrayBuffer();
                const decodedAudio = await audioContext.decodeAudioData(arrayBuffer);
                console.log(`Audio decoded. Sample rate: ${decodedAudio.sampleRate}, Length: ${decodedAudio.length}`);

                // Resample and convert to mono if necessary
                const processedAudio = await processAudio(decodedAudio, audioContext.sampleRate, EXPECTED_SAMPLE_RATE);
                console.log(`Audio processed. New length: ${processedAudio.length}`);

                if (processedAudio.length !== EXPECTED_AUDIO_LENGTH) {
                    statusElement.textContent = `Status: Error: Processed audio length mismatch. Expected ${EXPECTED_AUDIO_LENGTH}, got ${processedAudio.length}.`;
                    console.error('Processed audio length mismatch:', processedAudio.length);
                    return;
                }

                // --- DEBUG: Generate WAV for playback and download ---
                const wavBlob = encodeWAV(processedAudio, EXPECTED_SAMPLE_RATE);
                const audioUrl = URL.createObjectURL(wavBlob);
                audioPlayback.src = audioUrl;

                downloadLink.href = audioUrl;
                downloadLink.download = `debug_audio_${new Date().getTime()}.wav`;
                downloadLink.style.display = 'block';
                downloadLink.textContent = `Download Processed WAV (${processedAudio.length} samples)`;
                // ---------------------------------------------------

                // Run inference with WASM
                await runInference(processedAudio);
            } catch (err) {
                console.error('Error processing audio:', err);
                statusElement.textContent = `Status: Error processing audio: ${err.message}`;
            } finally {
                // Stop all tracks in the stream to release mic
                stream.getTracks().forEach(track => track.stop());
                startRecordingButton.disabled = false;
                stopRecordingButton.disabled = true;
            }
        };

        mediaRecorder.start();
        console.log('Recording started');
        statusElement.textContent = `Status: Recording for ${RECORDING_DURATION_MS / 1000} seconds...`;
        startRecordingButton.disabled = true;
        stopRecordingButton.disabled = false;

        // Stop recording automatically after duration
        setTimeout(() => {
            if (mediaRecorder.state === 'recording') {
                mediaRecorder.stop();
                console.log('Recording stopped via timeout');
            }
        }, RECORDING_DURATION_MS);

    } catch (e) {
        statusElement.textContent = `Status: Error accessing microphone: ${e.message}`;
        console.error('Error accessing microphone:', e);
        startRecordingButton.disabled = false;
        stopRecordingButton.disabled = true;
    }
}

async function processAudio(audioBuffer, originalSampleRate, targetSampleRate) {
    // 1. Resample usando OfflineAudioContext (Alta Calidad)
    // Esto evita el aliasing y la pérdida de agudos de la interpolación lineal
    let data;

    if (originalSampleRate !== targetSampleRate) {
        const offlineCtx = new OfflineAudioContext(
            1, // mono
            audioBuffer.duration * targetSampleRate,
            targetSampleRate
        );

        const source = offlineCtx.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(offlineCtx.destination);
        source.start();

        const resampledBuffer = await offlineCtx.startRendering();
        data = resampledBuffer.getChannelData(0);
    } else {
        // Si ya es 16k, solo tomamos el canal 0 (o mezclamos)
        data = audioBuffer.getChannelData(0);
        // Si fuera estéreo, habría que mezclar, pero OfflineCtx ya lo haría si se usara siempre
        if (audioBuffer.numberOfChannels > 1) {
            const monoData = new Float32Array(audioBuffer.length);
            for (let i = 0; i < audioBuffer.length; i++) {
                monoData[i] = (audioBuffer.getChannelData(0)[i] + audioBuffer.getChannelData(1)[i]) / 2;
            }
            data = monoData;
        }
    }

    // 2. High-Pass Filter (Simple IIR ~80Hz @ 16kHz)
    // Reactivado para eliminar DC offset y ruido grave que sesga hacia 'o'/'u'
    const alpha = 0.97;
    const filteredData = new Float32Array(data.length);
    filteredData[0] = data[0];
    for (let i = 1; i < data.length; i++) {
        filteredData[i] = alpha * (filteredData[i - 1] + data[i] - data[i - 1]);
    }
    data = filteredData;

    // 3. Smart Centering / Cropping
    if (data.length > EXPECTED_AUDIO_LENGTH) {
        const scanWindow = Math.floor(targetSampleRate * 0.05); // 50ms
        let maxEnergy = 0;
        let bestCenterIndex = Math.floor(data.length / 2);

        for (let i = 0; i < data.length - scanWindow; i += 100) {
            let currentEnergy = 0;
            for (let j = 0; j < scanWindow; j++) {
                currentEnergy += Math.abs(data[i + j]);
            }
            if (currentEnergy > maxEnergy) {
                maxEnergy = currentEnergy;
                bestCenterIndex = i + (scanWindow / 2);
            }
        }

        let startIndex = Math.floor(bestCenterIndex - (EXPECTED_AUDIO_LENGTH / 2));
        startIndex = Math.max(0, Math.min(startIndex, data.length - EXPECTED_AUDIO_LENGTH));
        data = data.slice(startIndex, startIndex + EXPECTED_AUDIO_LENGTH);
    } else if (data.length < EXPECTED_AUDIO_LENGTH) {
        const totalPadding = EXPECTED_AUDIO_LENGTH - data.length;
        const padLeft = Math.floor(totalPadding / 2);
        const padded = new Float32Array(EXPECTED_AUDIO_LENGTH).fill(0);
        padded.set(data, padLeft);
        data = padded;
    }

    // 4. Peak Normalization (Ahora sí, suave)
    // Para asegurar que la amplitud sea comparable al entrenamiento
    let maxVal = 0;
    for (let i = 0; i < data.length; i++) {
        if (Math.abs(data[i]) > maxVal) maxVal = Math.abs(data[i]);
    }
    if (maxVal > 0) {
        // Escalamos a 0.8 para dejar headroom (Python suele estar normalizado a +/- 1.0)
        // Evitamos escalar ruido de fondo (solo si hay señal decente)
        if (maxVal > 0.1) {
            const scale = 0.8 / maxVal;
            for (let i = 0; i < data.length; i++) {
                data[i] *= scale;
            }
        }
    }

    return data;
}

async function runInference(audioData) {
    statusElement.textContent = 'Status: Running WASM inference...';
    vowelElement.textContent = '';
    genderElement.textContent = '';

    try {
        const rawResult = await predict_vowel(audioData, EXPECTED_SAMPLE_RATE);
        const prediction = JSON.parse(rawResult);

        vowelElement.textContent = prediction.vowel;
        genderElement.textContent = prediction.gender;
        statusElement.textContent = 'Status: Inference complete. Ready to record.';

        console.log('Prediction:', prediction);

    } catch (e) {
        statusElement.textContent = `Status: Error during WASM inference: ${e.message}`;
        console.error('Error during WASM inference:', e);
    }
}

// --- WAV Helper Functions ---
function encodeWAV(samples, sampleRate) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    /* RIFF identifier */
    writeString(view, 0, 'RIFF');
    /* file length */
    view.setUint32(4, 36 + samples.length * 2, true);
    /* RIFF type */
    writeString(view, 8, 'WAVE');
    /* format chunk identifier */
    writeString(view, 12, 'fmt ');
    /* format chunk length */
    view.setUint32(16, 16, true);
    /* sample format (raw) */
    view.setUint16(20, 1, true);
    /* channel count */
    view.setUint16(22, 1, true);
    /* sample rate */
    view.setUint32(24, sampleRate, true);
    /* byte rate (sample rate * block align) */
    view.setUint32(28, sampleRate * 2, true);
    /* block align (channel count * bytes per sample) */
    view.setUint16(32, 2, true);
    /* bits per sample */
    view.setUint16(34, 16, true);
    /* data chunk identifier */
    writeString(view, 36, 'data');
    /* data chunk length */
    view.setUint32(40, samples.length * 2, true);

    floatTo16BitPCM(view, 44, samples);

    return new Blob([view], { type: 'audio/wav' });
}

function floatTo16BitPCM(output, offset, input) {
    for (let i = 0; i < input.length; i++, offset += 2) {
        let s = Math.max(-1, Math.min(1, input[i]));
        s = s < 0 ? s * 0x8000 : s * 0x7FFF;
        output.setInt16(offset, s, true);
    }
}

function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
    }
}
// ---------------------------

startRecordingButton.addEventListener('click', startRecording);
stopRecordingButton.addEventListener('click', stopRecording);

// Disable buttons until model is loaded
startRecordingButton.disabled = true;
stopRecordingButton.disabled = true;

loadModel();

// Basic styling (Vite's default creates an app.css)
import './style.css'