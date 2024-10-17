$(document).ready(function() {
    $.ajax ({
        url: '/gun_map',
        type: 'GET',
        success: function(data) {
            var plot_data = JSON.parse(data);
            Plotly.newPlot('map', plot_data);
        },
        error: function(error) {
            console.log('Error loading map:', error);
        }
    });
});

// document.addEventListener("DOMContentLoaded", function() {
//     // Retrieve the map data that is embedded in the HTML template
//     var plot_data = JSON.parse(document.getElementById('graph-data').textContent);

//     // Render the Plotly map in the 'map' div
//     Plotly.newPlot('map', plot_data.data, plot_data.layout);
// });