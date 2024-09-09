$(document).ready(function() {
    const totalImages = 9;
    let currentImage = localStorage.getItem('currentLogo');

    if (!currentImage || currentImage >= totalImages) {
        currentImage = 1;
    } else {
        currentImage = parseInt(currentImage) + 1;
    }
    logoSrc = `images/logo/${currentImage}.jpg`;

    // let logoSrc;

    // if (window.location.pathname.includes('game')) {
    //     logoSrc = `../../static/images/logo/${currentImage}.jpg`;
    // } else {
    //     logoSrc = `images/logo/${currentImage}.jpg`;
    // }

    $('#logo').attr('src', logoSrc);
    localStorage.setItem('currentLogo', currentImage);

    $('#logo').css({
        'opacity': '0',
        'transform': 'translateY(-100px)',
        'transition': 'none'
    });

    setTimeout(function() {
        $('#logo').css({
            'transition': 'transform 0.5s ease-out, opacity 1.3s ease-out',
            'transform': 'translateY(0)',
            'opacity': '1'
        });
    }, 100);

    $('#logo').on('click', function(event) {
        event.preventDefault();
        const link = $(this).closest('a').attr('href');

        $(this).css({
            'transform': 'translateY(100vh)',
            'transition': 'transform 0.5s ease-out'
        });

        $('body').addClass('fade-out');

        setTimeout(function() {
            window.location.href = link;
        }, 420);
    });
});