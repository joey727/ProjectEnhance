// Toast notification functionality
function showToast(type, title, message) {
    const toast = document.getElementById('toast');
    const toastIcon = document.getElementById('toast-icon');
    const toastTitle = document.getElementById('toast-title');
    const toastMessage = document.getElementById('toast-message');

    // Reset classes
    toast.classList.remove('toast-success', 'toast-error', 'toast-info');
    toastIcon.classList.remove('bg-green-50', 'bg-red-50', 'bg-blue-50');
    toastIcon.querySelector('svg').classList.remove('text-green-500', 'text-red-500', 'text-blue-500');

    // Set content
    toastTitle.textContent = title;
    toastMessage.textContent = message;

    // Set type-specific styling
    switch(type) {
        case 'success':
            toast.classList.add('toast-success');
            toastIcon.classList.add('bg-green-50');
            toastIcon.querySelector('svg').classList.add('text-green-500');
            toastIcon.innerHTML = `
                <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
                </svg>
            `;
            break;
        case 'error':
            toast.classList.add('toast-error');
            toastIcon.classList.add('bg-red-50');
            toastIcon.querySelector('svg').classList.add('text-red-500');
            toastIcon.innerHTML = `
                <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
                </svg>
            `;
            break;
        case 'info':
            toast.classList.add('toast-info');
            toastIcon.classList.add('bg-blue-50');
            toastIcon.querySelector('svg').classList.add('text-blue-500');
            toastIcon.innerHTML = `
                <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
            `;
            break;
    }

    // Show toast
    toast.classList.remove('hide');
    toast.classList.add('show');

    // Hide after 3 seconds
    setTimeout(() => {
        hideToast();
    }, 3000);
}

function hideToast() {
    const toast = document.getElementById('toast');
    toast.classList.remove('show');
    toast.classList.add('hide');
}

// Image upload handling
let fileInput, uploadArea, previewArea, originalPreview, enhancedPreview;

function initializeImageUpload() {
    fileInput = document.getElementById('file-upload');
    uploadArea = document.getElementById('upload-area');
    previewArea = document.getElementById('preview-area');
    originalPreview = document.getElementById('original-preview');
    enhancedPreview = document.getElementById('enhanced-preview');

    if (fileInput) {
        fileInput.addEventListener('change', handleFileSelect);
    }
}

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            originalPreview.src = e.target.result;
            uploadArea.classList.add('hidden');
            previewArea.classList.remove('hidden');
        };
        reader.readAsDataURL(file);
    }
}

async function processImage() {
    const file = fileInput.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);
    formData.append('enhancement_type', document.getElementById('enhancement-type').value);

    try {
        showToast('info', 'Processing...', 'Enhancing your image, please wait');
        
        const response = await fetch('/api/enhance', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('Enhancement failed');

        const result = await response.json();
        
        // Update enhanced preview
        enhancedPreview.innerHTML = `
            <img src="${result.enhanced_url}" class="w-full h-48 object-contain" alt="Enhanced image">
        `;

        // Show result actions
        document.getElementById('result-actions').classList.remove('hidden');
        document.getElementById('credits-used').textContent = result.credits_used || '1';

        // Show success toast
        showToast('success', 'Enhancement Complete!', 'Your image has been successfully enhanced');

    } catch (error) {
        console.error('Error:', error);
        showToast('error', 'Enhancement Failed', 'Failed to enhance image. Please try again.');
    }
}

function downloadImage() {
    const enhancedImage = document.querySelector('#enhanced-preview img');
    if (!enhancedImage) return;

    const link = document.createElement('a');
    link.href = enhancedImage.src;
    link.download = 'enhanced-image.png';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

function shareImage() {
    const enhancedImage = document.querySelector('#enhanced-preview img');
    if (!enhancedImage) return;

    if (navigator.share) {
        navigator.share({
            title: 'Enhanced Image',
            text: 'Check out my enhanced image!',
            url: enhancedImage.src
        }).catch(console.error);
    } else {
        showToast('info', 'Share Image', 'Copy the image URL to share');
    }
}

function resetUpload() {
    fileInput.value = '';
    uploadArea.classList.remove('hidden');
    previewArea.classList.add('hidden');
    document.getElementById('result-actions').classList.add('hidden');
    originalPreview.src = '';
    enhancedPreview.innerHTML = `
        <div class="text-center">
            <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
            <p class="text-sm text-slate-500 mt-2">Enhancing...</p>
        </div>
    `;
}

function showLogoutToast(event) {
    event.preventDefault();
    showToast('info', 'Logging out...', 'You are being signed out of your account');
    
    // Redirect after showing the toast
    setTimeout(() => {
        window.location.href = '/logout';
    }, 1000);
}

// Initialize everything when the DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeImageUpload();

    // Check if we're coming from a logout
    if (window.location.search.includes('logout=true')) {
        showToast('info', 'Logged Out', 'You have been successfully signed out');
    }

    // Add event listeners for buttons
    const logoutButton = document.querySelector('button[onclick="showLogoutToast(event)"]');
    if (logoutButton) {
        logoutButton.addEventListener('click', showLogoutToast);
    }

    const enhanceButton = document.querySelector('button[onclick="processImage()"]');
    if (enhanceButton) {
        enhanceButton.addEventListener('click', processImage);
    }

    const downloadButton = document.querySelector('button[onclick="downloadImage()"]');
    if (downloadButton) {
        downloadButton.addEventListener('click', downloadImage);
    }

    const shareButton = document.querySelector('button[onclick="shareImage()"]');
    if (shareButton) {
        shareButton.addEventListener('click', shareImage);
    }

    const resetButton = document.querySelector('button[onclick="resetUpload()"]');
    if (resetButton) {
        resetButton.addEventListener('click', resetUpload);
    }
});