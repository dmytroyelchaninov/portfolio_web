$('a.header__link').on('click', function(event) {
    event.preventDefault();
    const link = $(this).attr('href');
    
    // Add class to fade out
    $('body').addClass('fade-out');
  
    // Wait for the fade-out to complete before navigating
    setTimeout(function() {
      window.location.href = link;
    }, 200); // Match the fade-out transition time
  });