import pandas as pd
import plotly
import plotly.graph_objects as go
import json

def map():
    df = pd.read_csv('gun_violence/stage.csv')
    df_loc = df[['latitude', 'longitude', 'state', 'n_killed', 'n_injured', 'date', 'notes']]
    df_loc.dropna(inplace=True)

    colors = []
    for killed, injured in zip(df_loc['n_killed'], df_loc['n_injured']):
        if killed == 0 and injured == 0:
            colors.append('#43C13E')
        elif killed == 0 and injured > 0:
            colors.append('#3e43c1')
        else:
            colors.append('#C13E43')

    fig = go.Figure(go.Scattermapbox(
        lat=df_loc['latitude'],
        lon=df_loc['longitude'],
        mode='markers',
        marker=go.scattermapbox.Marker(size=9, color=colors),
        text=[f"<br>{note}<br>State: {state}<br>Killed:{killed}<br>Injured: {injured}<br>Date: {date}"
              for note, state, killed, injured, date in zip(df_loc['notes'], df_loc['state'], df_loc['n_killed'], df_loc['n_injured'], df_loc['date'])]
    ))

    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, mapbox_style="carto-positron", showlegend=False)

    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return graph_json
