import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from photonic_crystal2 import Crystal2D, CrystalSlab  # assuming the provided script is named photonic_crystal2.py

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Create the layout
app.layout = dbc.Container([
    html.H1("Photonic Crystal Simulator"),
    
    # Dropdown for selecting the photonic crystal type
    dbc.Row([
        dbc.Col(html.Label("Select Photonic Crystal Type")),
        dbc.Col(dcc.Dropdown(
            id='crystal-type-dropdown',
            options=[
                {'label': '2D Photonic Crystal', 'value': '2d'},
                {'label': 'Photonic Crystal Slab', 'value': 'slab'},  # Future option
            ],
            value='2d',  # Default to 2D photonic crystal
        )),
    ], className="mt-4"),

    # Configurator box (with a black border)
    dbc.Row([
        dbc.Col(html.Div(id='configurator-box', style={'border': '2px solid black', 'padding': '10px'})),
    ], className="mt-4"),

    # Button to run the simulation
    dbc.Row([
        dbc.Col(dbc.Button("Run Simulation", id="run-simulation-button", color="primary", className="mr-2"), width={"size": 3}),
    ], className="mt-3"),

    # Placeholder for results (two plots: epsilon on the left and bands on the right)
    dbc.Row([
        dbc.Col(dcc.Graph(id='epsilon-graph', style={'height': '700px', 'width': '700px'}), width=6),
        dbc.Col(dcc.Graph(id='bands-graph', style={'height': '700px', 'width': '700px'}, clickData=None), width=6),
    ], className="mt-4"),

    # Placeholder for field plots (TE-like on the left and TM-like on the right)
    dbc.Row([
        dbc.Col(dcc.Graph(id='te-field-graph', style={'height': '700px', 'width': '700px'}), width=6),
        dbc.Col(dcc.Graph(id='tm-field-graph', style={'height': '700px', 'width': '700px'}), width=6),
    ], className="mt-4")
])

# Callback to update the configurators based on the selected photonic crystal type
@app.callback(
    Output('configurator-box', 'children'),
    [Input('crystal-type-dropdown', 'value')]
)
def update_configurators(crystal_type):
    if crystal_type == '2d':
        return [
            html.Label("Lattice Type"),
            dcc.Dropdown(
                id='lattice-type-dropdown',
                options=[
                    {'label': 'Square', 'value': 'square'},
                    {'label': 'Triangular', 'value': 'triangular'}
                ],
                value='square',  # Default to square
            ),
            html.Label("Radius"),
            dcc.Input(id='radius-input', type='number', value=0.35, step=0.01),
            html.Label("Epsilon (Bulk Material)"),
            dcc.Input(id='epsilon-bulk-input', type='number', value=12, step=0.1),
            html.Label("Epsilon (Atom)"),
            dcc.Input(id='epsilon-atom-input', type='number', value=1, step=0.1)
        ]
    elif crystal_type == 'slab':
        # Placeholder for future configurators for crystal slab
        return html.Div("Crystal slab configurators are not implemented yet.")
    else:
        return html.Div("Invalid crystal type selected.")

# Callback to run the simulation and update the epsilon and band diagrams
@app.callback(
    [Output('epsilon-graph', 'figure'),
     Output('bands-graph', 'figure')],
    [Input('run-simulation-button', 'n_clicks')],
    [State('crystal-type-dropdown', 'value'),
     State('lattice-type-dropdown', 'value'),
     State('radius-input', 'value'),
     State('epsilon-bulk-input', 'value'),
     State('epsilon-atom-input', 'value')]
)
def run_simulation(n_clicks, crystal_type, lattice_type, radius, epsilon_bulk, epsilon_atom):
    if n_clicks is None:
        # Return empty figures if no simulation is run yet
        return go.Figure(), go.Figure()

    # Check the crystal type and run the respective simulation
    if crystal_type == '2d':
        # Initialize the 2D photonic crystal with the current configuration
        geometry = Crystal2D.basic_geometry(radius_1=radius, eps_atom_1=epsilon_atom, eps_bulk=epsilon_bulk)
        crystal = Crystal2D(lattice_type=lattice_type, geometry=geometry)

        # Set up and run the simulation for 2D photonic crystal
        crystal.set_solver()
        
        # Run both TM and TE simulations
        crystal.run_simulation("run_tm")
        crystal.run_simulation("run_te")
        crystal.extract_data()

        # Generate epsilon plot
        epsilon_fig = crystal.plot_epsilon_interactive()
        epsilon_fig.update_layout(width=700, height=700)  # Set figure size to 700x700

        # Generate band plot and plot both TE and TM bands
        bands_fig = crystal.plot_bands_interactive(polarization="tm", color="blue")  # Plot TM bands first
        crystal.plot_bands_interactive(polarization="te", color="red", fig=bands_fig)  # Overlay TE bands in red

        bands_fig.update_layout(width=700, height=700)  # Set figure size to 700x700

        return epsilon_fig, bands_fig

    elif crystal_type == 'slab':
        # Placeholder for Crystal Slab simulation (currently not implemented)
        placeholder_fig = go.Figure().update_layout(title="Crystal slab simulation is not implemented yet.", width=700, height=700)
        return placeholder_fig, placeholder_fig

    else:
        # Invalid crystal type, return empty figures
        empty_fig = go.Figure().update_layout(title="Invalid crystal type selected.", width=700, height=700)
        return empty_fig, empty_fig


# Callback to update the field plots based on the point clicked in the band diagram
@app.callback(
    [Output('te-field-graph', 'figure'),
     Output('tm-field-graph', 'figure')],
    [Input('bands-graph', 'clickData')],
    [State('crystal-type-dropdown', 'value'),
     State('lattice-type-dropdown', 'value'),
     State('radius-input', 'value'),
     State('epsilon-bulk-input', 'value'),
     State('epsilon-atom-input', 'value')]
)
def update_field_plots(clickData, crystal_type, lattice_type, radius, epsilon_bulk, epsilon_atom):
    if clickData is None:
        # Return empty figures if no point is clicked
        return go.Figure(), go.Figure()

    # Extract the clicked point information from clickData
    point_info = clickData['points'][0]
    k_index = point_info['x']  # Get the index of the k-point
    frequency = point_info['y']  # Get the frequency of the clicked band

    # Check the crystal type and generate the respective field plots
    if crystal_type == '2d':
        # Initialize the 2D photonic crystal with the current configuration
        geometry = Crystal2D.basic_geometry(radius_1=radius, eps_atom_1=epsilon_atom, eps_bulk=epsilon_bulk)
        crystal = Crystal2D(lattice_type=lattice_type, geometry=geometry)

        # Set up the solver with the specific k-point
        k_point = crystal.k_points_interpolated[k_index]
        crystal.set_solver(k_point=k_point)

        # Generate the field plot for TE-like modes
        te_field_fig = crystal.plot_field_interactive(runner="run_te", k_point=k_point)
        te_field_fig.update_layout(width=700, height=700)  # Set figure size to 700x700

        # Generate the field plot for TM-like modes
        tm_field_fig = crystal.plot_field_interactive(runner="run_tm", k_point=k_point)
        tm_field_fig.update_layout(width=700, height=700)  # Set figure size to 700x700

        return te_field_fig, tm_field_fig

    elif crystal_type == 'slab':
        # Placeholder for Crystal Slab field plots (currently not implemented)
        placeholder_fig = go.Figure().update_layout(title="Crystal slab fields are not implemented yet.", width=700, height=700)
        return placeholder_fig, placeholder_fig

    else:
        # Invalid crystal type, return empty figures
        empty_fig = go.Figure().update_layout(title="Invalid crystal type selected.", width=700, height=700)
        return empty_fig, empty_fig


# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
