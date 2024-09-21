document.addEventListener('DOMContentLoaded', function () {
    const fileInput = document.getElementById('file-input');
    const fileInputCarousel = document.getElementById('file-input-carousel');
    const imageWrapper = document.getElementById('image-wrapper');
    const carousel = document.getElementById('carousel');
    const carouselImages = document.querySelector('.carousel-images');
    const carouselProcessed = document.getElementById('carousel-processed');
    const carouselInitial = document.getElementById('carousel-initial');
    const dartboard = document.getElementById('dartboard');
    const prevButton = document.querySelector('.carousel-control.prev');
    const nextButton = document.querySelector('.carousel-control.next');
    const notificationProcess = document.querySelector('.notification-process');
    const notificationError = document.querySelector('.notification-error');
    const arrowContainer = document.querySelector('.arrow-container');
    const explanationText = document.querySelector('.explanation__text');
    const explanationArrow = document.querySelector('.explanation__arrow');
    const testButton = document.querySelector('.test-button');
    const randomImages = [ ];
    for (let i = 1; i<=10; i++) {
        randomImages.push(`static/test_images/${i}.jpg`);
    }

    explanationArrow.addEventListener('click', () => {
        explanationText.classList.toggle('expanded');
        explanationArrow.classList.toggle('expanded');
        testButton.classList.toggle('expanded');
    });
    
    let rotationInterval;
    let currentRotation = 0;
    const initialImageUrl = dartboard.src;
    let isProcessing = false;

    // startRotation(2000);

    imageWrapper.addEventListener('dragover', (event) => {
        event.preventDefault();
        imageWrapper.classList.add('drag-over');
    });

    imageWrapper.addEventListener('dragleave', () => {
        imageWrapper.classList.remove('drag-over');
    });

    imageWrapper.addEventListener('drop', (event) => {
        event.preventDefault();
        imageWrapper.classList.remove('drag-over');
        const files = event.dataTransfer.files;
        if (isProcessing) {
            notificationProcess.classList.add('show');
            console.log('Processing image. Please wait.');
            setTimeout(() => {
                notificationProcess.classList.remove('show');
            }, 3000);
            return;
        }
        if (files.length > 0) {
            fileInput.files = files;
            resetToInitialImage();
            startRotation();
            uploadImage(files[0]);
        }
    });

    imageWrapper.addEventListener('click', () => {
        fileInput.click();
    });

    carousel.addEventListener('click', () => {
        fileInputCarousel.click();
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            stopRotation();
            resetToInitialImage();
            startRotation();
            uploadImage(fileInput.files[0]);
        }
    });

    carousel.addEventListener('dragover', (event) => {
        event.preventDefault();
        carousel.classList.add('drag-over');
    });

    carousel.addEventListener('dragleave', () => {
        carousel.classList.remove('drag-over');
    });

    carousel.addEventListener('drop', (event) => {
        event.preventDefault();
        carousel.classList.remove('drag-over');
        const files = event.dataTransfer.files;
        if (files.length > 0) {
            fileInputCarousel.files = files;
            resetCarousel();
            resetToInitialImage();
            startRotation();
            uploadImage(files[0]);
        }
    });

    function startRotation(time=100) {
        if (!rotationInterval) {
            console.log("Starting rotation...");
            rotationInterval = setInterval(() => {
                currentRotation += 1;
                dartboard.style.transition = 'transform 0.1s linear';
                dartboard.style.transform = `rotate(${currentRotation}deg)`;
            }, time); // Rotate every 100 milliseconds
        }
    }

    function stopRotation() {
        // console.log(`rotationInterval: ${rotationInterval}`); // Debugging log
        // console.log(`currentRotation: ${currentRotation}`); // Debugging log
        if (rotationInterval) {
            console.log("Stopping rotation..."); // Debugging log
            clearInterval(rotationInterval);
            rotationInterval = null; // Clear the interval reference
        }
        dartboard.style.transform = `rotate(${currentRotation}deg)`;
    }

    function resetCarousel() {
        carousel.style.display = 'none';
        carouselProcessed.src = '';
        carouselInitial.src = '';
        currentRotation = 0;
        carouselImages.style.transform = 'translateX(0)';
        hideArrows();
        showImageWrapper();
    }

    function resetToInitialImage() {
        dartboard.src = initialImageUrl;
        dartboard.style.transform = '';
        carousel.style.display = 'none';
        hideArrows();
        showImageWrapper();
        startRotation();
    }

    function showImageWrapper() {
        imageWrapper.style.display = 'flex';
    }
    function showArrows() {
        prevButton.style.opacity = '1';
        prevButton.style.visibility = 'visible';
        nextButton.style.opacity = '1';
        nextButton.style.visibility = 'visible';
    }
    function hideArrows() {
        prevButton.style.opacity = '0';
        prevButton.style.visibility = 'hidden';
        nextButton.style.opacity = '0';
        nextButton.style.visibility = 'hidden';
    }

    function uploadImage(file) {
        isProcessing = true;
    
        const formData = new FormData();
        formData.append('file', file);
        
        const xhr = new XMLHttpRequest();
        xhr.open('POST', '/process_image', true);
        
        xhr.onload = function () {
            if (xhr.status === 200) {
                stopRotation();
                const response = JSON.parse(xhr.responseText);
                dartboard.src = response.image_url;    
                setTimeout(() => {
                    imageWrapper.style.display = 'none';
                    carouselProcessed.src = response.image_url;
                    carouselInitial.src = URL.createObjectURL(file);
                    carousel.style.display = 'block';
                    showArrows();
                    isProcessing = false;
                }, 0);
            } else {
                console.log(xhr.responseText);
                console.log(xhr.status);
                console.error('An error occurred while processing the image.');
                notificationError.classList.add('show');
                setTimeout(() => {
                    notificationError.classList.remove('show');
                }, 3000);
                stopRotation();
                isProcessing = false;
            }
        };
        
        xhr.onerror = function () {
            console.error('An error occurred while uploading the image.');
            notificationError.classList.add('show');
            setTimeout(() => {
                notificationError.classList.remove('show');
            }, 3000);
            stopRotation();
            isProcessing = false;
        };
        
        xhr.send(formData);
    }

    let currentSlide = 0;

    function showSlide(index) {
        const slides = document.querySelectorAll('.carousel-image');
        const totalSlides = slides.length;
    
        if (index >= totalSlides) {
            currentSlide = 0;
        } else if (index < 0) {
            currentSlide = totalSlides - 1;
        } else {
            currentSlide = index;
        }
    
        const offset = -currentSlide * 100;
        document.querySelector('.carousel-images').style.transform = `translateX(${offset}%)`;
    }
    window.prevSlide = function () {
        showSlide(currentSlide - 1);
    };
    window.nextSlide = function () {
        showSlide(currentSlide + 1);
    };
    let startX = 0;
    let currentX = 0;
    let isSwiping = false;

    carousel.addEventListener('touchstart', (event) => {
        startX = event.touches[0].clientX;
        isSwiping = true;
    });
    carousel.addEventListener('touchmove', (event) => {
        if (!isSwiping) return;
        currentX = event.touches[0].clientX;
    });
    carousel.addEventListener('touchend', () => {
        const swipeDistance = startX - currentX;
        
        if (swipeDistance > 50) {
            // Swipe left: show the next slide
            nextSlide();
        } else if (swipeDistance < -50) {
            // Swipe right: show the previous slide
            prevSlide();
        }

        isSwiping = false;
    });

    // Function to simulate a random image being "dropped"
    function testImage() {
        // if (rotationInterval) {
        //     console.log("Stopping rotation...");
        //     clearInterval(rotationInterval);
        //     rotationInterval = null;
        // }
        // dartboard.style.transform = `rotate(${currentRotation}deg)`;

        if (isProcessing) {
            console.log("Image processing is in progress. Please wait.");
            notificationProcess.style.display = 'flex';
            setTimeout(() => {
                notificationProcess.style.display = 'none';
            }, 3000);
            return;
        }
    
        // Select a random image from the array
        const randomIndex = Math.floor(Math.random() * randomImages.length);
        const selectedImage = randomImages[randomIndex];
    
        // Simulate file input change and trigger the upload process
        fetch(selectedImage)
            .then(response => response.blob())
            .then(blob => {
                const file = new File([blob], `testImage.jpg`, { type: blob.type });
                resetToInitialImage();
                setTimeout(() => {
                    resetCarousel();
                    resetToInitialImage();
                    startRotation();
                    uploadImage(file);
                }, 0);
            })
            .catch(err => {
                console.error('An error occurred while fetching the test image:', err);
                notificationError.classList.add('show');
                setTimeout(() => {
                    notificationError.classList.remove('show');
                }, 3000);
            });
    }
    window.testImage = testImage;
});