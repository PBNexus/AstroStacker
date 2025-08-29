class AstroStacker {
    constructor() {
        this.sessionId = this.generateSessionId();
        this.lightFiles = [];
        this.darkFiles = [];
        this.biasFiles = []; // NEW: Bias files array
        this.flatFiles = []; // NEW: Flat files array
        this.selectedOutputFormat = 'png'; // Default output format
        this.selectedBackgroundSubtractionMethod = 'median'; // NEW: Default background subtraction method
        this.outputFileUrl = ''; // To store the URL of the final stacked image
        this.previewFileUrl = ''; // To store the URL of the preview image
        this.init();
    }

    generateSessionId() {
        return Date.now().toString(36) + Math.random().toString(36).substr(2);
    }

    init() {
        this.setupEventListeners();
        // Changed version to v0.4
        this.logToConsole('🚀 AstroStacker v0.4 initialized. Ready for stellar stacking!');
    }

    // Helper function to safely get elements and log if not found
    getElement(id) {
        const element = document.getElementById(id);
        if (!element) {
            // Log to console directly, avoiding circular dependency with logToConsole for this specific case
            console.error(`[ERROR] Element with ID '${id}' not found. This might prevent some functionality.`);
        }
        return element;
    }

    setupEventListeners() {
        // Setup for drag and drop zones
        const lightDropZone = this.getElement('lightDropZone');
        if (lightDropZone) this.setupDropZone(lightDropZone, 'light');

        const darkDropZone = this.getElement('darkDropZone');
        if (darkDropZone) this.setupDropZone(darkDropZone, 'dark');

        const biasDropZone = this.getElement('biasDropZone');
        if (biasDropZone) this.setupDropZone(biasDropZone, 'bias');

        const flatDropZone = this.getElement('flatDropZone');
        if (flatDropZone) this.setupDropZone(flatDropZone, 'flat');

        // Setup for "Browse" button clicks
        const lightBrowseBtn = this.getElement('lightBrowseBtn');
        if (lightBrowseBtn) lightBrowseBtn.addEventListener('click', () => this.getElement('lightFileInput').click());

        const darkBrowseBtn = this.getElement('darkBrowseBtn');
        if (darkBrowseBtn) darkBrowseBtn.addEventListener('click', () => this.getElement('darkFileInput').click());

        const biasBrowseBtn = this.getElement('biasBrowseBtn'); // NEW
        if (biasBrowseBtn) biasBrowseBtn.addEventListener('click', () => this.getElement('biasFileInput').click());

        const flatBrowseBtn = this.getElement('flatBrowseBtn'); // NEW
        if (flatBrowseBtn) flatBrowseBtn.addEventListener('click', () => this.getElement('flatFileInput').click());

        // Setup for file input changes (when files are selected via browse dialog)
        const lightFileInput = this.getElement('lightFileInput');
        if (lightFileInput) lightFileInput.addEventListener('change', (event) => this.handleFiles(event.target.files, 'light'));

        const darkFileInput = this.getElement('darkFileInput');
        if (darkFileInput) darkFileInput.addEventListener('change', (event) => this.handleFiles(event.target.files, 'dark'));

        const biasFileInput = this.getElement('biasFileInput'); // NEW
        if (biasFileInput) biasFileInput.addEventListener('change', (event) => this.handleFiles(event.target.files, 'bias'));

        const flatFileInput = this.getElement('flatFileInput'); // NEW
        if (flatFileInput) flatFileInput.addEventListener('change', (event) => this.handleFiles(event.target.files, 'flat'));

        // Setup for Stack and Clear buttons
        const stackBtn = this.getElement('stackBtn');
        if (stackBtn) stackBtn.addEventListener('click', () => this.startStackingProcess());

        const clearBtn = this.getElement('clearBtn');
        if (clearBtn) clearBtn.addEventListener('click', () => this.clearAll());

        // Setup for output format and background subtraction method selection
        const outputFormat = this.getElement('outputFormat');
        if (outputFormat) {
            outputFormat.addEventListener('change', (event) => {
                this.selectedOutputFormat = event.target.value;
                this.logToConsole(`Output format set to: ${this.selectedOutputFormat.toUpperCase()}`);
            });
        }

        const backgroundSubtractionMethod = this.getElement('backgroundSubtractionMethod');
        if (backgroundSubtractionMethod) {
            backgroundSubtractionMethod.addEventListener('change', (event) => {
                this.selectedBackgroundSubtractionMethod = event.target.value;
                this.logToConsole(`Background subtraction method set to: ${this.selectedBackgroundSubtractionMethod.charAt(0).toUpperCase() + this.selectedBackgroundSubtractionMethod.slice(1)}`);
            });
        }

        // Event listener for the download button
        const downloadBtn = this.getElement('downloadBtn');
        if (downloadBtn) {
            downloadBtn.addEventListener('click', (event) => {
                event.preventDefault(); 
                if (this.outputFileUrl) {
                    const a = document.createElement('a');
                    a.href = this.outputFileUrl;
                    a.download = this.outputFileUrl.split('/').pop(); 
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    this.logToConsole('Download initiated.');
                } else {
                    this.logToConsole('No file available for download.', 'warning');
                }
            });
        }
    }

    setupDropZone(dropZone, type) {
        dropZone.addEventListener('dragover', (event) => {
            event.preventDefault();
            dropZone.classList.add('drag-over');
        });

        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('drag-over');
        });

        dropZone.addEventListener('drop', (event) => {
            event.preventDefault();
            dropZone.classList.remove('drag-over');
            this.handleFiles(event.dataTransfer.files, type);
        });
    }

    handleFiles(files, type) {
        if (files.length === 0) return;

        let fileListElement; 
        switch (type) {
            case 'light': this.lightFiles = Array.from(files); fileListElement = this.getElement('lightFileList'); break;
            case 'dark': this.darkFiles = Array.from(files); fileListElement = this.getElement('darkFileList'); break;
            case 'bias': this.biasFiles = Array.from(files); fileListElement = this.getElement('biasFileList'); break;
            case 'flat': this.flatFiles = Array.from(files); fileListElement = this.getElement('flatFileList'); break;
            default: return;
        }

        if (fileListElement) { 
            fileListElement.innerHTML = ''; 
            Array.from(files).forEach(file => {
                const listItem = document.createElement('div');
                listItem.textContent = file.name;
                fileListElement.appendChild(listItem);
            });
        }
        this.logToConsole(`Added ${files.length} ${type} frame(s).`);
    }

    async startStackingProcess() {
        this.logToConsole('Initiating stacking process...');
        this.setProgress(0, 'Starting upload...', true); // Show glow for upload

        // Hide previous results and warnings
        this.getElement('resultSection').style.display = 'none';
        this.getElement('validationWarningsSection').style.display = 'none';
        this.getElement('stackedImagePreview').style.display = 'none';
        this.getElement('downloadBtn').style.display = 'none'; // Hide download button initially

        if (this.lightFiles.length === 0) {
            this.logToConsole('Please upload at least one light frame.', 'error');
            this.setProgress(0, 'Stacking failed: No light frames.', false);
            return;
        }

        try {
            // 1. Upload all files
            const allFiles = [
                ...this.lightFiles.map(f => ({ file: f, type: 'light' })),
                ...this.darkFiles.map(f => ({ file: f, type: 'dark' })),
                ...this.biasFiles.map(f => ({ file: f, type: 'bias' })),
                ...this.flatFiles.map(f => ({ file: f, type: 'flat' }))
            ];

            const formData = new FormData();
            formData.append('session_id', this.sessionId);
            allFiles.forEach(item => {
                // CORRECTED: Strip existing type prefix from filename before re-applying the correct one
                const originalFileName = item.file.name;
                const typePrefix = `${item.type}_`;
                let cleanedFileName = originalFileName;

                // Check if the filename already starts with a known type prefix and strip it
                // This handles cases where files might be dragged from previous sessions or downloads
                // that already have a prefix like "light_IMG_0702.CR2"
                if (cleanedFileName.startsWith('light_')) {
                    cleanedFileName = cleanedFileName.substring('light_'.length);
                } else if (cleanedFileName.startsWith('dark_')) {
                    cleanedFileName = cleanedFileName.substring('dark_'.length);
                } else if (cleanedFileName.startsWith('bias_')) {
                    cleanedFileName = cleanedFileName.substring('bias_'.length);
                } else if (cleanedFileName.startsWith('flat_')) {
                    cleanedFileName = cleanedFileName.substring('flat_'.length);
                }
                
                // Now append with the correct prefix for the current item's type
                // This ensures the filename sent to the backend is always type_originalName.ext
                formData.append('files', item.file, `${item.type}_${cleanedFileName}`); 
            });

            this.logToConsole('Uploading files to server...');
            const uploadResponse = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            if (!uploadResponse.ok) {
                const errorData = await uploadResponse.json();
                throw new Error(`Upload failed: ${errorData.detail || uploadResponse.statusText}`);
            }
            this.logToConsole('Files uploaded successfully.');
            this.setProgress(20, 'Files uploaded, validating metadata...');


            // 2. Validate metadata
            this.logToConsole('Validating metadata...');
            // Send empty string for optional parameters if their file lists are empty
            const validateFormData = new FormData();
            validateFormData.append('session_id', this.sessionId);
            // Append filenames with their type prefixes, as they are now saved on the backend
            this.lightFiles.map(f => `light_${f.name}`).forEach(name => validateFormData.append('light_files', name));
            
            // For optional lists, append all names if present, or an empty string if empty
            if (this.darkFiles.length > 0) {
                this.darkFiles.map(f => `dark_${f.name}`).forEach(name => validateFormData.append('dark_files', name));
            } else {
                validateFormData.append('dark_files', ''); // Send empty string
            }
            if (this.biasFiles.length > 0) {
                this.biasFiles.map(f => `bias_${f.name}`).forEach(name => validateFormData.append('bias_files', name));
            } else {
                validateFormData.append('bias_files', ''); // Send empty string
            }
            if (this.flatFiles.length > 0) {
                this.flatFiles.map(f => `flat_${f.name}`).forEach(name => validateFormData.append('flat_files', name));
            } else {
                validateFormData.append('flat_files', ''); // Send empty string
            }

            const validateResponse = await fetch('/validate', {
                method: 'POST',
                body: validateFormData 
            });

            if (!validateResponse.ok) {
                const errorData = await validateResponse.json();
                throw new Error(`Validation failed: ${errorData.detail || validateResponse.statusText}`);
            }
            const validationResult = await validateResponse.json();
            this.displayValidationWarnings(validationResult.validation_warnings);
            this.logToConsole('Metadata validation complete.');
            this.setProgress(40, 'Metadata validated, starting stacking...');


            // 3. Start stacking
            this.logToConsole('Sending stacking request...');
            // Send empty string for optional parameters if their file lists are empty
            const stackFormData = new FormData();
            stackFormData.append('session_id', this.sessionId);
            // Append filenames with their type prefixes, as they are now saved on the backend
            this.lightFiles.map(f => `light_${f.name}`).forEach(name => stackFormData.append('light_files', name));
            
            // For optional lists, append all names if present, or an empty string if empty
            if (this.darkFiles.length > 0) {
                this.darkFiles.map(f => `dark_${f.name}`).forEach(name => stackFormData.append('dark_files', name));
            }
            if (this.biasFiles.length > 0) {
                this.biasFiles.map(f => `bias_${f.name}`).forEach(name => stackFormData.append('bias_files', name));
            } else {
                stackFormData.append('bias_files', '');
            }
            if (this.flatFiles.length > 0) {
                this.flatFiles.map(f => `flat_${f.name}`).forEach(name => stackFormData.append('flat_files', name));
            } else {
                stackFormData.append('flat_files', '');
            }
            stackFormData.append('output_format', this.selectedOutputFormat);
            stackFormData.append('background_subtraction_method', this.selectedBackgroundSubtractionMethod);


            const stackResponse = await fetch('/stack', {
                method: 'POST',
                body: stackFormData
            });

            if (!stackResponse.ok) {
                const errorData = await stackResponse.json();
                throw new Error(`Stacking failed: ${errorData.detail || stackResponse.statusText}`);
            }

            const stackResult = await stackResponse.json();
            this.handleStackResponse(stackResult);

        } catch (error) {
            this.logToConsole(`Stacking process failed: ${error.message}`, 'error');
            this.setProgress(0, 'Stacking failed ❌', false);
        }
    }

    handleStackResponse(result) {
        this.logToConsole(result.message, 'success');
        this.setProgress(100, 'Stacking complete! ✨', false);

        const resultSection = this.getElement('resultSection');
        const resultInfo = this.getElement('resultInfo');
        const stackedImagePreview = this.getElement('stackedImagePreview');
        const downloadBtn = this.getElement('downloadBtn');

        if (resultInfo) resultInfo.textContent = `Your image has been stacked successfully!`;
        if (resultSection) resultSection.style.display = 'block';

        if (result.preview_url) {
            this.previewFileUrl = result.preview_url;
            if (stackedImagePreview) {
                stackedImagePreview.src = result.preview_url;
                stackedImagePreview.style.display = 'block';
            }
            this.logToConsole(`Preview available at: ${result.preview_url}`);
        } else {
            this.logToConsole('No preview image URL received.', 'warning');
            if (stackedImagePreview) stackedImagePreview.style.display = 'none';
        }

        if (result.download_url) {
            this.outputFileUrl = result.download_url;
            if (downloadBtn) {
                downloadBtn.href = result.download_url;
                downloadBtn.download = result.download_url.split('/').pop(); 
                downloadBtn.style.display = 'block';
            }
            this.logToConsole(`Download link available for: ${result.download_url.split('/').pop()}`);
        } else {
            this.logToConsole('No download URL received.', 'warning');
            if (downloadBtn) downloadBtn.style.display = 'none';
        }
    }

    displayValidationWarnings(warnings) {
        const warningsSection = this.getElement('validationWarningsSection');
        const warningsList = this.getElement('warningsList');
        
        if (warningsList) warningsList.innerHTML = ''; // Clear previous warnings

        if (warnings && warnings.length > 0) {
            if (warningsSection) warningsSection.style.display = 'block';
            warnings.forEach(warning => {
                const warningLine = document.createElement('div');
                warningLine.className = `console-line ${warning.level}`; // Using console-line for consistent styling
                warningLine.textContent = `[${warning.level.toUpperCase()}] ${warning.message}`;
                if (warningsList) warningsList.appendChild(warningLine);
            });
        } else {
            if (warningsSection) warningsSection.style.display = 'none';
        }
    }

    clearAll() {
        this.lightFiles = [];
        this.darkFiles = [];
        this.biasFiles = [];
        this.flatFiles = [];
        this.outputFileUrl = '';
        this.previewFileUrl = '';

        // Clear file lists in UI
        const lightFileList = this.getElement('lightFileList');
        if (lightFileList) lightFileList.innerHTML = '';
        const darkFileList = this.getElement('darkFileList');
        if (darkFileList) darkFileList.innerHTML = '';
        const biasFileList = this.getElement('biasFileList');
        if (biasFileList) biasFileList.innerHTML = '';
        const flatFileList = this.getElement('flatFileList');
        if (flatFileList) flatFileList.innerHTML = '';

        // Reset progress bar
        this.setProgress(0, 'Ready to stack images ✨', false);

        // Hide result and warnings sections
        const resultSection = this.getElement('resultSection');
        if (resultSection) resultSection.style.display = 'none';
        const validationWarningsSection = this.getElement('validationWarningsSection');
        if (validationWarningsSection) validationWarningsSection.style.display = 'none';
        const stackedImagePreview = this.getElement('stackedImagePreview');
        if (stackedImagePreview) stackedImagePreview.style.display = 'none';
        const downloadBtn = this.getElement('downloadBtn');
        if (downloadBtn) downloadBtn.style.display = 'none'; // Hide download button on clear

        this.logToConsole('🧹 Cleared all files and reset the application.');
        // Clear console output
        const consoleEl = document.getElementById('console'); // Direct access here
        if (consoleEl) {
            consoleEl.innerHTML = `
                <div class="console-line">🌌 AstroStacker v0.4 initialized</div>
                <div class="console-line">Ready for some stellar image stacking...</div>
            `;
        }
    }

    setProgress(percentage, text, showGlow = false) {
        const progressFill = this.getElement('progressFill');
        const progressText = this.getElement('progressText');
        
        if (progressFill) progressFill.style.width = percentage + '%';
        if (progressText) progressText.textContent = text;

        if (progressFill) {
            if (showGlow) {
                progressFill.classList.add('glow');
            } else {
                progressFill.classList.remove('glow');
            }
        }
    }

    logToConsole(message, level = 'info') {
        const consoleEl = document.getElementById('console'); // Direct access here
        if (!consoleEl) {
            console.error(`[CRITICAL ERROR] Console element not found! Cannot log message: ${message}`);
            return;
        }
        const line = document.createElement('div');
        line.className = `console-line ${level}`;
        
        const timestamp = new Date().toLocaleTimeString();
        line.textContent = `[${timestamp}] ${message}`;
        
        consoleEl.appendChild(line);
        // Automatically scroll to the latest message
        consoleEl.scrollTop = consoleEl.scrollHeight;
    }
}

// Initialize the application once the DOM is fully loaded
window.addEventListener('DOMContentLoaded', () => {
    const astroStacker = new AstroStacker();
});
