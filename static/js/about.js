$('.dots').on('click', function() {
    const id = $(this).attr('id');
    
    if (id=== "dots0") {
        $('#dots0').hide();
        $('#more0').show();
    }

    if (id === "dots1") {
        $('#dots1').hide();
        $('#more1').show();
        $('#dots2').show();
    } else if (id === "dots2") {
        $('#dots2').hide();
        $('#more2').show();
    }
});