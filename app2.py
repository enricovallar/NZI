import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import pickle
import tkinter as tk
from tkinter import filedialog
import base64
import io
import plotly.graph_objects as go
from photonic_crystal2 import Crystal2D, CrystalSlab  # assuming the provided script is named photonic_crystal2.py

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], prevent_initial_callbacks='initial_duplicate')
crystal_active = None 
configuration_active = None
active_crystal_has_been_run = False

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
        dbc.Col(html.Div([
            html.Div(id='configurator-box', style={'border': '2px solid black', 'padding': '10px'}),
        ])),
    ], className="mt-4"),

    # Buttons to save and load the crystal
    dbc.Row([
        dbc.Col(dcc.Upload(id='upload-crystal', children=dbc.Button("Load Crystal", color="secondary")), width={"size": 4}),
        dbc.Col(dbc.Button("Save Crystal", id="save-crystal-button", color="secondary"), width={"size": 4}),
        dbc.Col(dbc.Button("Update Crystal", id="update-crystal-button", color="secondary"), width={"size": 4}),
        dcc.Download(id="download-crystal")
    ], className="mt-3"),

    # Buttons to show the dielectric and run the simulation
    dbc.Row([
        dbc.Col(dbc.Button("Show Dielectric", id="show-dielectric-button", color="primary", className="mr-2"), width={"size": 4}),
        dbc.Col(dbc.Button("Run Simulation", id="run-simulation-button", color="primary", className="mr-2"), width={"size": 4}),
        dbc.Col(width={"size": 4}),  # Blank column
    ], className="mt-3"),

    # Scrollable text area for messages to the user
    dbc.Row([
        dbc.Col(html.Div([
            dcc.Textarea(
                id='message-box',
                value='',
                style={'width': '100%', 'height': '100px', 'overflowY': 'scroll'},
                readOnly=True
            )
        ]))
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


def switch_configurator(crystal_type, configuration=None):
    default_values = {
        'crystal_id': 'crystal_1',
        'lattice_type': 'square',
        'radius': 0.35,
        'epsilon_bulk': 12,
        'epsilon_atom': 1,
        'epsilon_background': 1,
        'height_slab': 0.5,
        'height_supercell': 4
    }

    if configuration is not None:
        default_values = configuration

    common_inputs = [
        dbc.Row([
            dbc.Col(html.Label("Crystal ID"), width=4),
            dbc.Col(dcc.Input(id='crystal-id-input', type='text', value=default_values['crystal_id'], placeholder='Enter Crystal ID'), width=8),
        ], style={'margin-bottom': '10px'}),
        
        dbc.Row([
            dbc.Col(html.Label("Lattice Type"), width=4),
            dbc.Col(dcc.Dropdown(
                id='lattice-type-dropdown',
                options=[
                    {'label': 'Square', 'value': 'square'},
                    {'label': 'Triangular', 'value': 'triangular'}
                ],
                value=default_values['lattice_type'],  # Default to square
            ), width=8),
        ], style={'margin-bottom': '10px'}),
        
        dbc.Row([
            dbc.Col(html.Label("Radius"), width=4),
            dbc.Col(dcc.Input(id='radius-input', type='number', value=default_values['radius'], step=0.01), width=8),
        ], style={'margin-bottom': '10px'}),
        
        dbc.Row([
            dbc.Col(html.Label("Epsilon (Bulk Material)"), width=4),
            dbc.Col(dcc.Input(id='epsilon-bulk-input', type='number', value=default_values['epsilon_bulk'], step=0.1), width=8),
        ], style={'margin-bottom': '10px'}),
        
        dbc.Row([
            dbc.Col(html.Label("Epsilon (Atom)"), width=4),
            dbc.Col(dcc.Input(id='epsilon-atom-input', type='number', value=default_values['epsilon_atom'], step=0.1), width=8),
        ], style={'margin-bottom': '10px'}),
    ]

    if crystal_type == '2d':
        return common_inputs + [
            dbc.Row([
                dbc.Col(html.Label("Epsilon (Background)"), width=4),
                dbc.Col(dcc.Input(id='epsilon-background-input', type='number', value=default_values['epsilon_background'], step=0.1), width=8),
            ], style={'display': 'none'}),  # Hide these inputs for 2D crystal
            
            dbc.Row([
                dbc.Col(html.Label("Height (Slab)"), width=4),
                dbc.Col(dcc.Input(id='height-slab-input', type='number', value=default_values['height_slab'], step=0.01), width=8),
            ], style={'display': 'none'}),  # Hide these inputs for 2D crystal
            
            dbc.Row([
                dbc.Col(html.Label("Height (Supercell)"), width=4),
                dbc.Col(dcc.Input(id='height-supercell-input', type='number', value=default_values['height_supercell'], step=0.1), width=8),
            ], style={'display': 'none'})  # Hide these inputs for 2D crystal
        ]
    elif crystal_type == 'slab':
        return common_inputs + [
            dbc.Row([
                dbc.Col(html.Label("Epsilon (Background)"), width=4),
                dbc.Col(dcc.Input(id='epsilon-background-input', type='number', value=default_values['epsilon_background'], step=0.1), width=8),
            ], style={'margin-bottom': '10px'}),
            
            dbc.Row([
                dbc.Col(html.Label("Height (Slab)"), width=4),
                dbc.Col(dcc.Input(id='height-slab-input', type='number', value=default_values['height_slab'], step=0.01), width=8),
            ], style={'margin-bottom': '10px'}),
            
            dbc.Row([
                dbc.Col(html.Label("Height (Supercell)"), width=4),
                dbc.Col(dcc.Input(id='height-supercell-input', type='number', value=default_values['height_supercell'], step=0.1), width=8),
            ], style={'margin-bottom': '10px'}),
        ]
    else:
        return html.Div("Invalid crystal type selected.")



# Callback to update the configurators based on the selected photonic crystal type
@app.callback(
    Output('configurator-box', 'children'),
    [Input('crystal-type-dropdown', 'value')]
)
def update_configurators(crystal_type):
    return switch_configurator(crystal_type)
    
# Callback to set the active crystal and configuration
@app.callback(
    Output('message-box', 'value', allow_duplicate=True),
    [Input('update-crystal-button', 'n_clicks')],
    [State('crystal-id-input', 'value'),
     State('crystal-type-dropdown', 'value'),
     State('lattice-type-dropdown', 'value'),
     State('radius-input', 'value'),
     State('epsilon-bulk-input', 'value'),
     State('epsilon-atom-input', 'value'),
     State('epsilon-background-input', 'value'),
     State('height-slab-input', 'value'),
     State('height-supercell-input', 'value')],
    prevent_initial_call=True
)
def update_crystal(n_clicks, crystal_id, crystal_type, lattice_type, radius, epsilon_bulk, epsilon_atom, epsilon_background, height_slab, height_supercell):
    if n_clicks is None:
        return ""

    global crystal_active, configuration_active, active_crystal_has_been_run

    if crystal_type == '2d':
        geometry = Crystal2D.basic_geometry(radius_1=radius, eps_atom_1=epsilon_atom, eps_bulk=epsilon_bulk)
        crystal_active = Crystal2D(lattice_type=lattice_type, geometry=geometry)
    elif crystal_type == 'slab':
        geometry = CrystalSlab.basic_geometry(radius_1=radius, eps_atom_1=epsilon_atom, eps_bulk=epsilon_bulk, eps_background=epsilon_background, height_slab=height_slab, height_supercell=height_supercell)
        crystal_active = CrystalSlab(lattice_type=lattice_type, geometry=geometry)
    else:
        return "Invalid crystal type selected."

    configuration_active = {
        'crystal_id': crystal_id,
        'crystal_type': crystal_type,
        'lattice_type': lattice_type,
        'radius': radius,
        'epsilon_bulk': epsilon_bulk,
        'epsilon_atom': epsilon_atom,
        'epsilon_background': epsilon_background,
        'height_slab': height_slab,
        'height_supercell': height_supercell
    }

    active_crystal_has_been_run = False
    return "The active crystal has been updated."

# Callback to pickle the active crystal and configuration and trigger a download.
@app.callback(
    [Output('download-crystal', 'data'),
     Output('message-box', 'value', allow_duplicate=True)],
    [Input('save-crystal-button', 'n_clicks')],
    [State('crystal-id-input', 'value'),
     State('crystal-type-dropdown', 'value'),
     State('lattice-type-dropdown', 'value'),
     State('radius-input', 'value'),
     State('epsilon-bulk-input', 'value'),
     State('epsilon-atom-input', 'value'),
     State('epsilon-background-input', 'value'),
     State('height-slab-input', 'value'),
     State('height-supercell-input', 'value')],
    prevent_initial_call=True
)
def save_crystal(n_clicks, crystal_id, crystal_type, lattice_type, radius, epsilon_bulk, epsilon_atom, epsilon_background, height_slab, height_supercell):
    if n_clicks is None:
        return None, ""

    global crystal_active, configuration_active, active_crystal_has_been_run

    if crystal_active is None:
        return dcc.send_string("No crystal has been created. Save operation aborted.", "error.txt"), "No crystal has been created. Save operation aborted."

    if not active_crystal_has_been_run:
        return dcc.send_string("The active crystal has not been run. Save operation aborted.", "error.txt"), "The active crystal has not been run. Save operation aborted."

    data = {
        'crystal': crystal_active,
        'configuration': configuration_active,
        'active_crystal_has_been_run': active_crystal_has_been_run
    }
    serialized_data = pickle.dumps(data)
    filename = f"{crystal_id}.pkl" if crystal_id else "crystal.pkl"

    return dcc.send_bytes(serialized_data, filename), f"Crystal configuration saved successfully as {filename}. Please choose where to download the file."


# Callback to load the active crystal and configuration using dcc.Upload.
@app.callback(
    [Output('message-box', 'value', allow_duplicate=True),
     Output('crystal-id-input', 'value', allow_duplicate=True),
     Output('crystal-type-dropdown', 'value', allow_duplicate=True),
     Output('lattice-type-dropdown', 'value', allow_duplicate=True),
     Output('radius-input', 'value', allow_duplicate=True),
     Output('epsilon-bulk-input', 'value', allow_duplicate=True),
     Output('epsilon-atom-input', 'value', allow_duplicate=True),
     Output('epsilon-background-input', 'value', allow_duplicate=True),
     Output('height-slab-input', 'value', allow_duplicate=True),
     Output('height-supercell-input', 'value', allow_duplicate=True),
     Output('configurator-box', 'children', allow_duplicate=True)],
    [Input('upload-crystal', 'contents')],
    [State('upload-crystal', 'filename')],
    prevent_initial_call=True
)
def load_crystal(contents, filename):
    if contents is None:
        return "", dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    global crystal_active, configuration_active, active_crystal_has_been_run

    content_string = contents.split(',')[1]
    decoded = base64.b64decode(content_string)
    data = pickle.loads(decoded)
    crystal_active = data['crystal']
    configuration_active = data['configuration']
    active_crystal_has_been_run = data['active_crystal_has_been_run']
    
    print("Loaded the following data")
    print(configuration_active)
    print(crystal_active)
    print(f"runned before?{active_crystal_has_been_run}")

    configurator_children = switch_configurator(configuration_active['crystal_type'], configuration_active)

    return (f"Crystal configuration loaded successfully from {filename}.",
            configuration_active['crystal_id'],
            configuration_active['crystal_type'],
            configuration_active['lattice_type'],
            configuration_active['radius'],
            configuration_active['epsilon_bulk'],
            configuration_active['epsilon_atom'],
            configuration_active['epsilon_background'],
            configuration_active['height_slab'],
            configuration_active['height_supercell'],
            configurator_children)

#function to be called when the plot epsilon button is clicked
def plot_epsilon(crystal_type, lattice_type, radius, epsilon_bulk, epsilon_atom, epsilon_background, height_slab, height_supercell):
    global crystal_active, configuration_active

    if crystal_active is None:
        if crystal_type == '2d':
            geometry = Crystal2D.basic_geometry(radius_1=radius, eps_atom_1=epsilon_atom, eps_bulk=epsilon_bulk)
            crystal_active = Crystal2D(lattice_type=lattice_type, geometry=geometry)
            # Update the active configuration
            configuration_active = {
                'crystal_id': 'crystal_1',  # Update this with the actual crystal ID if available
                'crystal_type': crystal_type,
                'lattice_type': lattice_type,
                'radius': radius,
                'epsilon_bulk': epsilon_bulk,
                'epsilon_atom': epsilon_atom,
                'epsilon_background': epsilon_background,
                'height_slab': height_slab,
                'height_supercell': height_supercell
            }
        elif crystal_type == 'slab':
            geometry = CrystalSlab.basic_geometry(radius_1=radius, 
                                                  eps_atom_1=epsilon_atom, 
                                                  eps_bulk=epsilon_bulk, 
                                                  eps_background=epsilon_background, 
                                                  height_slab=height_slab, 
                                                  height_supercell=height_supercell)
            crystal_active = CrystalSlab(lattice_type=lattice_type, geometry=geometry)
            # Update the active configuration
            configuration_active = {
                'crystal_id': 'crystal_1',  # Update this with the actual crystal ID if available
                'crystal_type': crystal_type,
                'lattice_type': lattice_type,
                'radius': radius,
                'epsilon_bulk': epsilon_bulk,
                'epsilon_atom': epsilon_atom,
                'epsilon_background': epsilon_background,
                'height_slab': height_slab,
                'height_supercell': height_supercell
            }
        else:
            return go.Figure()
        # Notify the user that the active crystal has changed
        print("The active crystal has been updated.")
         
    crystal_active.run_dumb_simulation()  # Run a dummy simulation to get the epsilon array
    crystal_active.extract_data()

    if crystal_type == '2d':
        epsilon_fig = crystal_active.plot_epsilon_interactive()
    elif crystal_type == 'slab':
        epsilon_fig = crystal_active.plot_epsilon_interactive(opacity=0.2, 
                                                              colorscale='matter', 
                                                              override_resolution_with=-1,
                                                              periods=2)
    else:
        return go.Figure()
    
    epsilon_fig.update_layout(width=700, height=700)
    print(crystal_active)
    return epsilon_fig



# Function to be called when the run_simulation is clicked
def run_simulation(crystal):
    

    if crystal is None:
        return go.Figure(), go.Figure(), "Please update the active crystal before running the simulation."

        

    crystal.set_solver()
    crystal.run_simulation("run_tm")
    crystal.run_simulation("run_te")
    crystal.extract_data()

    if isinstance(crystal, Crystal2D):
        epsilon_fig = crystal.plot_epsilon_interactive()
        epsilon_fig.update_layout(width=700, height=700)
        bands_fig = crystal.plot_bands_interactive(polarization="tm", color="blue")
        crystal.plot_bands_interactive(polarization="te", color="red", fig=bands_fig)
        bands_fig.update_layout(width=700, height=700)
    elif isinstance(crystal, CrystalSlab):
        epsilon_fig = crystal.plot_epsilon_interactive(opacity=0.2, colorscale='matter', override_resolution_with=-1, periods=2)
        epsilon_fig.update_layout(width=700, height=700)
        bands_fig = crystal.plot_bands_interactive(polarization="tm", color="blue")
        crystal.plot_bands_interactive(polarization="te", color="red", fig=bands_fig)
        bands_fig.update_layout(width=700, height=700)
    else:
        empty_fig = go.Figure().update_layout(title="Invalid crystal type selected.", width=700, height=700)
        return empty_fig, empty_fig
    
    global active_crystal_has_been_run
    active_crystal_has_been_run = True
    return epsilon_fig, bands_fig, "Simulation run completed."

# Combined callback to handle both "Run Simulation" and "Show Dielectric" buttons
@app.callback(
    [Output('epsilon-graph', 'figure', allow_duplicate=True),
     Output('bands-graph', 'figure', allow_duplicate=True),
     Output('message-box', 'value', allow_duplicate=True)],
    [Input('run-simulation-button', 'n_clicks'),
     Input('show-dielectric-button', 'n_clicks')],
    [State('crystal-type-dropdown', 'value'),
     State('lattice-type-dropdown', 'value'),
     State('radius-input', 'value'),
     State('epsilon-bulk-input', 'value'),
     State('epsilon-atom-input', 'value'),
     State('epsilon-background-input', 'value'),
     State('height-slab-input', 'value'),
     State('height-supercell-input', 'value')],
    prevent_initial_call=True
)
def handle_buttons(run_clicks, show_clicks, crystal_type, lattice_type, radius, epsilon_bulk, epsilon_atom, epsilon_background, height_slab, height_supercell):
    ctx = dash.callback_context
    global crystal_active

    if not ctx.triggered:
        return go.Figure(), go.Figure(), ""

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'show-dielectric-button':
        epsilon_fig = plot_epsilon(crystal_type, 
                                   lattice_type, 
                                   radius, 
                                   epsilon_bulk, 
                                   epsilon_atom, 
                                   epsilon_background, 
                                   height_slab, 
                                   height_supercell)

        return epsilon_fig, go.Figure(), "Dielectric plot generated."

    elif button_id == 'run-simulation-button':
        epsilon_fig, bands_fig, msg = run_simulation(crystal_active)
        
        return epsilon_fig, bands_fig, msg
        
# Callback to update the field plots based on the point clicked in the band diagram
@app.callback(
    [Output('te-field-graph', 'figure'),
     Output('tm-field-graph', 'figure')],
    [Input('bands-graph', 'clickData')],
    [State('crystal-type-dropdown', 'value'),
     State('lattice-type-dropdown', 'value'),
     State('radius-input', 'value'),
     State('epsilon-bulk-input', 'value'),
     State('epsilon-atom-input', 'value'),
     State('epsilon-background-input', 'value'),
     State('height-slab-input', 'value'),
     State('height-supercell-input', 'value')]
)
def update_field_plots(clickData, crystal_type, lattice_type, radius, epsilon_bulk, epsilon_atom, epsilon_background, height_slab, height_supercell):
    if clickData is None:
        return go.Figure(), go.Figure()

    # Extract the clicked point data
    point = clickData['points'][0]
    kx = point['x']
    ky = point['y']

    if crystal_type == '2d':
        geometry = Crystal2D.basic_geometry(radius_1=radius, eps_atom_1=epsilon_atom, eps_bulk=epsilon_bulk)
        crystal = Crystal2D(lattice_type=lattice_type, geometry=geometry)
        crystal.set_solver()
        crystal.run_simulation("run_tm")
        crystal.run_simulation("run_te")
        crystal.extract_data()
        te_field_fig = crystal.plot_field_interactive(kx=kx, ky=ky, polarization="te")
        tm_field_fig = crystal.plot_field_interactive(kx=kx, ky=ky, polarization="tm")
        return te_field_fig, tm_field_fig

    elif crystal_type == 'slab':
        geometry = CrystalSlab.basic_geometry(radius_1=radius, eps_atom_1=epsilon_atom, eps_bulk=epsilon_bulk, eps_background=epsilon_background, height_slab=height_slab, height_supercell=height_supercell)
        crystal = CrystalSlab(lattice_type=lattice_type, geometry=geometry)
        crystal.set_solver()
        crystal.run_simulation("run_tm")
        crystal.run_simulation("run_te")
        crystal.extract_data()
        te_field_fig = crystal.plot_field_interactive(kx=kx, ky=ky, polarization="te")
        tm_field_fig = crystal.plot_field_interactive(kx=kx, ky=ky, polarization="tm")
        return te_field_fig, tm_field_fig

    else:
        empty_fig = go.Figure().update_layout(title="Invalid crystal type selected.", width=700, height=700)
        return empty_fig, empty_fig

if __name__ == '__main__':
    app.run(debug=True)