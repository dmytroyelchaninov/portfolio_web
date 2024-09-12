$(document).ready(function() {
    const messageText = document.getElementById('messageText');
    const messageButton = document.getElementById('messageButton');

    // Add an event listener for focus
    messageText.addEventListener('focus', function() {
        // Change the background color when clicked (focused)
        messageText.style.backgroundColor = "#f4f4f4";  // New color on focus
        messageButton.style.backgroundColor = "#f4f4f4";  // New color on focus
    });
    
    // Optionally, you can add an event for when the textarea loses focus
    messageText.addEventListener('blur', function() {
        // Revert the background color when focus is lost
        messageText.style.backgroundColor = "";  // Revert to original color
        messageButton.style.backgroundColor = "";  // Revert to
    });
    $('#messageForm').on('submit', function(event) {
        event.preventDefault();
        
        const message = $('#messageText').val();

        $.ajax({
            url: '/send_message',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ message: message }),
            success: function(response) {
                console.log('Success');
                $('.contact_form').addClass('hide');
                $('#messageSuccess').css('display', 'flex');
            },
            error: function(response) {
                console.log('Error');
                $('.contact_form').addClass('hide');
                $('#messageError').css('display', 'flex');
            }
        })
    });
});