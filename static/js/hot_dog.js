document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.querySelector('#file-input');
    const dragdropImage = document.querySelector('#hot__dragdrop');
    const hotdogImage = document.querySelector('.hot__images_hotdog');
    const hotDescription = document.querySelector('.hot__description_text');
    const hotArrow = document.querySelector('.hot__description_arrow');
    const uploadedImage = document.getElementById('hot__uploaded-image');
    const uploadedImageWrapper = document.querySelector('.hot__images_uploaded-image');
    const processingText = document.querySelector('.hot__result_processing');
    const resultPos = document.querySelector('#hot__result-pos');
    const resultNeg = document.querySelector('#hot__result-neg');
    const resultText = document.querySelector('.hot__result_text');
    const notificationProcess = document.querySelector('.hot__notification-process');
    const notificationError = document.querySelector('.hot__notification-error');
    let isProcessing = false;

    // Rolling description text
    const toggleExpanded = () => {
        hotDescription.classList.toggle('expanded');
        hotArrow.classList.toggle('expanded');
    };

    hotArrow.addEventListener('click', toggleExpanded);
    hotArrow.addEventListener('touchstart', toggleExpanded); 

    // Drag and drop functionality for hotdog image
    hotdogImage.addEventListener('dragover', (event) => {
        event.preventDefault();
        hotdogImage.classList.add('drag-over');
    });

    hotdogImage.addEventListener('dragleave', () => {
        hotdogImage.classList.remove('drag-over');
    });

    hotdogImage.addEventListener('drop', (event) => {
        event.preventDefault();
        hotdogImage.classList.remove('drag-over');
        handleFileDrop(event);
    });

    uploadedImage.addEventListener('dragover', (event) => {
        event.preventDefault();
        uploadedImage.classList.add('drag-over');
    });

    uploadedImage.addEventListener('dragleave', () => {
        uploadedImage.classList.remove('drag-over');
    });

    uploadedImage.addEventListener('drop', (event) => {
        event.preventDefault();
        uploadedImage.classList.remove('drag-over');
        handleFileDrop(event);
    });
    hotdogImage.addEventListener('click', () => {
        fileInput.click();
    });

    function handleFileDrop(event) {
        const files = event.dataTransfer.files;
        if (isProcessing) {
            notificationProcess.classList.add('show');
            setTimeout(() => {
                notificationProcess.classList.remove('show');
            }, 3000);
            return;
        }

        if (files.length > 0) {
            fileInput.files = files;
            uploadImage(files[0]);
        }
    }

    fileInput.addEventListener('change', function() {
        const file = fileInput.files[0];
        if (file) {
            uploadImage(file);
        }
    });

    function uploadImage(file) {
        console.log("Uploading image...");
        const formData = new FormData();
        formData.append('file', file);
        isProcessing = true;

        fetch('/is_hot_dog', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            isProcessing = false;
            const imgURL = URL.createObjectURL(file);
            uploadedImage.src = imgURL;
            hotdogImage.style.display = 'none';
            uploadedImageWrapper.style.display = 'flex';
            
            if (data.result) {
                resultPos.style.display = 'flex';
                resultNeg.style.display = 'none';
            } else {
                resultNeg.style.display = 'flex';
                resultPos.style.display = 'none';
            }
        })
        .catch(err => {
            isProcessing = false;
            console.error('Error:', err);
            notificationError.classList.add('show');
            setTimeout(() => {
                notificationError.classList.remove('show');
            }, 3000);
        });
    }
});