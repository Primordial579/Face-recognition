class FaceRecognitionApp {
  constructor() {
    this.stream = null;
    this.isCapturing = false;
    this.isRegistering = false;
    this.isRecognizing = false;
    this.capturedImages = [];
    this.targetSamples = 50;
    this.apiBaseUrl = 'https://facerecog-1.onrender.com';
    
    this.initializeElements();
    this.bindEvents();
  }

  initializeElements() {
    // Tabs
    this.registerTab = document.getElementById('registerTab');
    this.loginTab = document.getElementById('loginTab');
    this.registerContent = document.getElementById('registerContent');
    this.loginContent = document.getElementById('loginContent');

    // Register
    this.nameInput = document.getElementById('nameInput');
    this.startCameraBtn = document.getElementById('startCameraBtn');
    this.cameraSection = document.getElementById('cameraSection');
    this.video = document.getElementById('video');
    this.captureOverlay = document.getElementById('captureOverlay');
    this.captureStatus = document.getElementById('captureStatus');
    this.progressSection = document.getElementById('progressSection');
    this.progressFill = document.getElementById('progressFill');
    this.captureBtn = document.getElementById('captureBtn');
    this.stopCameraBtn = document.getElementById('stopCameraBtn');
    this.registerStatus = document.getElementById('registerStatus');

    // Login
    this.startLoginCameraBtn = document.getElementById('startLoginCameraBtn');
    this.loginCameraSection = document.getElementById('loginCameraSection');
    this.loginVideo = document.getElementById('loginVideo');
    this.recognizeOverlay = document.getElementById('recognizeOverlay');
    this.recognizeBtn = document.getElementById('recognizeBtn');
    this.stopLoginCameraBtn = document.getElementById('stopLoginCameraBtn');
    this.recognitionResult = document.getElementById('recognitionResult');

    // Canvas
    this.canvas = document.getElementById('canvas');
    this.ctx = this.canvas.getContext('2d');
  }

  bindEvents() {
    this.registerTab.addEventListener('click', () => this.switchTab('register'));
    this.loginTab.addEventListener('click', () => this.switchTab('login'));

    this.startCameraBtn.addEventListener('click', () => this.startCamera('register'));
    this.captureBtn.addEventListener('click', () => this.handleCapture());
    this.stopCameraBtn.addEventListener('click', () => this.stopCamera());

    this.startLoginCameraBtn.addEventListener('click', () => this.startCamera('login'));
    this.recognizeBtn.addEventListener('click', () => this.handleRecognize());
    this.stopLoginCameraBtn.addEventListener('click', () => this.stopCamera());
  }

  switchTab(tab) {
    this.stopCamera();
    if (tab === 'register') {
      this.registerTab.classList.add('active');
      this.loginTab.classList.remove('active');
      this.registerContent.classList.remove('hidden');
      this.loginContent.classList.add('hidden');
    } else {
      this.loginTab.classList.add('active');
      this.registerTab.classList.remove('active');
      this.loginContent.classList.remove('hidden');
      this.registerContent.classList.add('hidden');
    }
  }

  async startCamera(mode) {
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
      if (mode === 'register') {
        this.video.srcObject = this.stream;
        this.startCameraBtn.classList.add('hidden');
        this.cameraSection.classList.remove('hidden');
      } else {
        this.loginVideo.srcObject = this.stream;
        this.startLoginCameraBtn.classList.add('hidden');
        this.loginCameraSection.classList.remove('hidden');
      }
      this.isCapturing = true;
    } catch (error) {
      console.error('Error accessing camera:', error);
      this.showStatus('error', 'Could not access camera. Please check permissions.');
    }
  }

  stopCamera() {
    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
      this.stream = null;
    }
    this.isCapturing = false;
    this.startCameraBtn.classList.remove('hidden');
    this.cameraSection.classList.add('hidden');
    this.captureOverlay.classList.add('hidden');
    this.progressSection.classList.add('hidden');

    this.startLoginCameraBtn.classList.remove('hidden');
    this.loginCameraSection.classList.add('hidden');
    this.recognizeOverlay.classList.add('hidden');
  }

  captureFrame(videoElement) {
    if (!videoElement || !this.canvas) return null;
    this.canvas.width = videoElement.videoWidth;
    this.canvas.height = videoElement.videoHeight;
    this.ctx.drawImage(videoElement, 0, 0);
    return this.canvas.toDataURL('image/jpeg', 0.8);
  }

  async handleCapture() {
    const name = this.nameInput.value.trim();
    if (!name) {
      this.showStatus('error', 'Please enter your name first');
      return;
    }
    this.isRegistering = true;
    this.capturedImages = [];
    this.captureBtn.disabled = true;
    this.stopCameraBtn.disabled = true;
    this.nameInput.disabled = true;
    this.captureOverlay.classList.remove('hidden');
    this.progressSection.classList.remove('hidden');
    this.showStatus('info', 'Capturing face samples...');
    const captureInterval = setInterval(() => {
      const frame = this.captureFrame(this.video);
      if (frame) {
        this.capturedImages.push(frame);
        this.updateProgress();
        if (this.capturedImages.length >= this.targetSamples) {
          clearInterval(captureInterval);
          this.submitRegistration(name);
        }
      }
    }, 100);
  }

  updateProgress() {
    const progress = Math.round((this.capturedImages.length / this.targetSamples) * 100);
    this.progressFill.style.width = `${progress}%`;
    this.captureStatus.textContent = `Capturing... ${this.capturedImages.length}/${this.targetSamples}`;
  }

  async submitRegistration(name) {
    try {
      this.showStatus('info', 'Processing and training model...');
      const response = await fetch(`${this.apiBaseUrl}/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: name, images: this.capturedImages })
      });
      const data = await response.json();
      if (data.success) {
        this.showStatus('success', `✅ Success: ${data.message}`);
        this.resetRegistration();
      } else {
        this.showStatus('error', `❌ Error: ${data.message}`);
      }
    } catch (error) {
      this.showStatus('error', '❌ Error: Could not connect to server');
    } finally {
      this.isRegistering = false;
      this.captureBtn.disabled = false;
      this.stopCameraBtn.disabled = false;
      this.nameInput.disabled = false;
      this.captureOverlay.classList.add('hidden');
      this.progressSection.classList.add('hidden');
    }
  }

  resetRegistration() {
    this.nameInput.value = '';
    this.capturedImages = [];
    this.progressFill.style.width = '0%';
    this.stopCamera();
  }

