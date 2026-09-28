/**
 * main.js - Frontend Audio Upload, Live Microphone Recorder & Dynamic Results Handler
 */

document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const audioInput = document.getElementById('audio-input');
    const browseBtn = document.getElementById('browse-btn');
    const recordBtn = document.getElementById('record-btn');
    const recordingStatus = document.getElementById('recording-status');
    const recTimer = document.getElementById('rec-timer');

    const audioPreviewContainer = document.getElementById('audio-preview-container');
    const fileNameDisplay = document.getElementById('file-name-display');
    const audioPlayer = document.getElementById('audio-player');
    const analyzeBtn = document.getElementById('analyze-btn');

    const emptyState = document.getElementById('empty-state');
    const loadingState = document.getElementById('loading-state');
    const resultsView = document.getElementById('results-view');

    const predictedClassText = document.getElementById('predicted-class-text');
    const confidenceText = document.getElementById('confidence-text');
    const segmentCountText = document.getElementById('segment-count-text');
    const progressList = document.getElementById('progress-list');
    const disclaimerText = document.getElementById('disclaimer-text');

    let selectedFile = null;
    let isLiveSelected = false;
    let isRecording = false;
    let mediaRecorder = null;
    let audioChunks = [];
    let recordCountdownInterval = null;

    // Trigger File Input Click
    browseBtn.addEventListener('click', () => audioInput.click());
    dropZone.addEventListener('click', (e) => {
        if (e.target !== browseBtn && !browseBtn.contains(e.target)) {
            audioInput.click();
        }
    });

    // Drag & Drop Handlers
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
        });
    });

    dropZone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelection(files[0]);
        }
    });

    audioInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelection(e.target.files[0]);
        }
    });

    function handleFileSelection(file, isLiveRecord = false) {
        if (!file.name.toLowerCase().endsWith('.wav') && !isLiveRecord) {
            alert('Please select a valid .wav audio file.');
            return;
        }

        selectedFile = file;
        isLiveSelected = isLiveRecord;
        fileNameDisplay.textContent = file.name;
        
        const audioUrl = URL.createObjectURL(file);
        audioPlayer.src = audioUrl;

        audioPreviewContainer.classList.remove('hidden');

        // Animate wave visualizer when audio is played
        const waveBars = document.querySelectorAll('.wave-bar');
        audioPlayer.onplay = () => {
            waveBars.forEach(bar => bar.style.animationPlayState = 'running');
        };
        audioPlayer.onpause = () => {
            waveBars.forEach(bar => bar.style.animationPlayState = 'paused');
        };
        waveBars.forEach(bar => bar.style.animationPlayState = 'paused');
    }

    // =========================================================================
    // Live Microphone Recording Handler (HTML5 MediaRecorder + PCM WAV Encoder)
    // =========================================================================
    recordBtn.addEventListener('click', async () => {
        if (isRecording) return;

        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            startLiveRecording(stream);
        } catch (err) {
            alert('Microphone access denied or not supported by browser: ' + err.message);
        }
    });

    function startLiveRecording(stream) {
        isRecording = true;
        audioChunks = [];

        recordBtn.classList.add('recording');
        recordBtn.disabled = true;
        recordBtn.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i> Recording Live Sound...';
        recordingStatus.classList.remove('hidden');

        let secondsLeft = 5;
        recTimer.textContent = `${secondsLeft}s`;

        recordCountdownInterval = setInterval(() => {
            secondsLeft -= 1;
            recTimer.textContent = `${secondsLeft}s`;
            if (secondsLeft <= 0) {
                clearInterval(recordCountdownInterval);
            }
        }, 1000);

        mediaRecorder = new MediaRecorder(stream);
        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = async () => {
            stream.getTracks().forEach(track => track.stop());
            const rawBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType });
            
            // Convert WebM/OGG stream to standard 16kHz WAV format using Web Audio API
            try {
                const wavBlob = await convertBlobToWav(rawBlob);
                const liveWavFile = new File([wavBlob], 'Live_Cough_Recording.wav', { type: 'audio/wav' });
                handleFileSelection(liveWavFile, true);
            } catch (err) {
                console.error('WAV encoding error:', err);
                const liveWavFile = new File([rawBlob], 'Live_Cough_Recording.wav', { type: rawBlob.type });
                handleFileSelection(liveWavFile, true);
            }

            resetRecordButtonUI();
        };

        mediaRecorder.start();

        // Automatically stop recording after exactly 5.0 seconds
        setTimeout(() => {
            if (mediaRecorder && mediaRecorder.state === 'recording') {
                mediaRecorder.stop();
            }
        }, 5000);
    }

    function resetRecordButtonUI() {
        isRecording = false;
        clearInterval(recordCountdownInterval);
        recordBtn.classList.remove('recording');
        recordBtn.disabled = false;
        recordBtn.innerHTML = '<i class="fa-solid fa-microphone"></i> Record Live Cough Sound (5s)';
        recordingStatus.classList.add('hidden');
    }

    // Convert raw browser audio recording into PCM 16-bit Mono WAV format
    async function convertBlobToWav(blob) {
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const arrayBuffer = await blob.arrayBuffer();
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

        return audioBufferToWavBlob(audioBuffer);
    }

    function audioBufferToWavBlob(buffer) {
        const numChannels = 1;
        const sampleRate = buffer.sampleRate;
        const format = 1; // PCM
        const bitDepth = 16;
        
        let samples = buffer.getChannelData(0);
        let numSamples = samples.length;
        let bytesPerSample = bitDepth / 8;
        let blockAlign = numChannels * bytesPerSample;
        let byteRate = sampleRate * blockAlign;
        let dataSize = numSamples * blockAlign;
        let bufferSize = 44 + dataSize;
        
        let arrayBuffer = new ArrayBuffer(bufferSize);
        let view = new DataView(arrayBuffer);
        
        writeString(view, 0, 'RIFF');
        view.setUint32(4, 36 + dataSize, true);
        writeString(view, 8, 'WAVE');
        writeString(view, 12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, format, true);
        view.setUint16(22, numChannels, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, byteRate, true);
        view.setUint16(32, blockAlign, true);
        view.setUint16(34, bitDepth, true);
        writeString(view, 36, 'data');
        view.setUint32(40, dataSize, true);
        
        let offset = 44;
        for (let i = 0; i < samples.length; i++, offset += 2) {
            let s = Math.max(-1, Math.min(1, samples[i]));
            view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
        }
        
        return new Blob([view], { type: 'audio/wav' });
    }

    function writeString(view, offset, string) {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    }

    // Submit Audio File to Flask Backend (/predict)
    analyzeBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        emptyState.classList.add('hidden');
        resultsView.classList.add('hidden');
        loadingState.classList.remove('hidden');

        // Reset and start step ticker animation sequence
        const steps = [
            document.getElementById('step-1'),
            document.getElementById('step-2'),
            document.getElementById('step-3'),
            document.getElementById('step-4')
        ];
        
        steps.forEach(s => {
            if (s) {
                s.classList.remove('active', 'completed');
            }
        });
        if (steps[0]) steps[0].classList.add('active');

        let currentStep = 0;
        const stepInterval = setInterval(() => {
            if (currentStep < steps.length - 1) {
                if (steps[currentStep]) {
                    steps[currentStep].classList.remove('active');
                    steps[currentStep].classList.add('completed');
                }
                currentStep++;
                if (steps[currentStep]) {
                    steps[currentStep].classList.add('active');
                }
            }
        }, 600);

        const formData = new FormData();
        formData.append('audio', selectedFile);
        formData.append('is_live', isLiveSelected ? 'true' : 'false');

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            clearInterval(stepInterval);

            if (!response.ok || data.error) {
                alert(data.error || 'An error occurred during audio processing.');
                loadingState.classList.add('hidden');
                emptyState.classList.remove('hidden');
                return;
            }

            // Short pause to complete visual step feedback
            setTimeout(() => {
                renderResults(data);
            }, 500);

        } catch (err) {
            clearInterval(stepInterval);
            alert('Network or server connection failed: ' + err.message);
            loadingState.classList.add('hidden');
            emptyState.classList.remove('hidden');
        }
    });

    function renderResults(data) {
        loadingState.classList.add('hidden');
        resultsView.classList.remove('hidden');

        predictedClassText.textContent = data.prediction;
        confidenceText.textContent = `${data.confidence.toFixed(1)}%`;
        segmentCountText.textContent = data.segment_count;

        progressList.innerHTML = '';
        data.breakdown.forEach(item => {
            const isWinner = item.class === data.prediction;

            const itemEl = document.createElement('div');
            itemEl.className = 'progress-item';

            itemEl.innerHTML = `
                <div class="progress-labels">
                    <span style="font-weight: ${isWinner ? '600' : '400'}; color: ${isWinner ? '#06b6d4' : '#cbd5e1'}">
                        ${isWinner ? '<i class="fa-solid fa-crown" style="color: #f59e0b; margin-right: 4px;"></i>' : ''}${item.class}
                    </span>
                    <span>${item.confidence.toFixed(1)}%</span>
                </div>
                <div class="progress-bar-bg">
                    <div class="progress-bar-fill" style="width: 0%; ${isWinner ? 'background: linear-gradient(90deg, #3b82f6, #06b6d4);' : 'background: rgba(255,255,255,0.2);'}"></div>
                </div>
            `;

            progressList.appendChild(itemEl);

            setTimeout(() => {
                const fill = itemEl.querySelector('.progress-bar-fill');
                fill.style.width = `${item.confidence}%`;
            }, 50);
        });
    }
});
