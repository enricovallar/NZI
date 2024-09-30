#%%
import math
import meep as mp
from meep import mpb
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
from photonic_crystal import Crystal2D, CrystalSlab  # assuming the provided script is named photonic_crystal2.py
from ui_elements import *
import dash_daq as daq


def string_to_vector3(vector_string):
    try:
        # Remove the parentheses and split the string by commas
        vector_components = vector_string.strip('()').split(',')
        # Convert the components to floats and create an mp.Vector3 object
        return mp.Vector3(float(vector_components[0]), float(vector_components[1]), float(vector_components[2]))
    except (ValueError, IndexError) as e:
        raise ValueError(f"Invalid vector string format: {vector_string}") from e
    


# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], prevent_initial_callbacks='initial_duplicate')
crystal_active = None 
configuration_active = None


# Create the layout
app.layout = dbc.Container([
    html.H1("Photonic Crystal Simulator"),
    
    # Configurator box (with a black border)
    dbc.Row([

        # Box on the left to set up geometry properties
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.H4("Geometry Configuration"),
                    html.Div(id='geometry-configurator-box', children=[
                        dbc.Row(
                            geometry_configuration_elements_list,
                            className="mt-4"),
                    ])
                ]),
                style={'border': '2px solid black', 'padding': '10px', 'height': '100%', 'overflowY': 'scroll'}
            ),
            width=4, style={'height': '400px'}
        ),

        # Box in the middle to set up material properties
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.H4("Material Configuration"),
                    dbc.Row(
                        [
                            dbc.Col(html.Label("Advanced Material"), width=4),
                            dbc.Col(daq.BooleanSwitch(id='advanced-material-toggle', on=False), width=8),
                        ],
                        style={'padding': '10px', "display": "flex"},
                        id='advanced-material-toggle-box'
                    ),
                    html.Div(id='material-configurator-box', children=[
                        dbc.Row(
                            material_configuration_elements_list,
                            className="mt-4"),
                    ])
                ]),
                style={'border': '2px solid black', 'padding': '10px', 'height': '100%', 'overflowY': 'scroll'}
            ),
            width=4, style={'height': '400px'}
        ),

        # Box on the right to set up solver properties
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.H4("Solver Configuration"),
                    html.Div(id='solver-configurator-box', children=[
                        dbc.Row(
                            solver_configuration_elements_list,
                            className="mt-4"),
                    ])
                ]),
                style={'border': '2px solid black', 'padding': '10px', 'height': '100%', 'overflowY': 'scroll'}
            ),
            width=4, style={'height': '400px'}
        )
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
        dbc.Col(dcc.Graph(id='epsilon-graph', style={'height': '700px', 'width': '700px', 'padding-right': '200px'}), width=6),
        dbc.Col(dcc.Graph(id='bands-graph', style={'height': '700px', 'width': '700px', 'padding-left': '200px'}, clickData=None), width=6),
    ], className="mt-4"),

    # Placeholder for field plots (TE-like on the left and TM-like on the right)
    dbc.Row([
        dbc.Col(dcc.Graph(id='te-field-graph', style={'height': '700px', 'width': '700px', 'padding-right': '200px'}), width=6),
        dbc.Col(dcc.Graph(id='tm-field-graph', style={'height': '700px', 'width': '700px', 'padding-left': '200px'}), width=6),
    ], className="mt-4")


])

# callback to update the configurator-box content when a different photonic crystal type is selected
@app.callback(
    [Output('geometry-configurator-box', 'children'),
     Output('material-configurator-box', 'children'),
     Output('solver-configurator-box', 'children'),
     Output('crystal-type-dropdown', 'value')],
    Input('crystal-type-dropdown', 'value')
)
def update_configurator(crystal_type):
    global crystal_active, configuration_active

    if crystal_type == '2d':
        material_configuration_elements['epsilon-background-input'].hide()
        geometry_configuration_elements['height-slab-input'].hide()
        geometry_configuration_elements['height-supercell-input'].hide()
        solver_configuration_elements['resolution-2d-input'].show()
        solver_configuration_elements['resolution-3d-input'].hide()

        return geometry_configuration_elements_list, material_configuration_elements_list, solver_configuration_elements_list, crystal_type
          
    elif crystal_type == 'slab':
        material_configuration_elements['epsilon-background-input'].show()
        geometry_configuration_elements['height-slab-input'].show()
        geometry_configuration_elements['height-supercell-input'].show()
        solver_configuration_elements['resolution-2d-input'].hide()
        solver_configuration_elements['resolution-3d-input'].show()
        return geometry_configuration_elements_list, material_configuration_elements_list, solver_configuration_elements_list, crystal_type
    else:
        return [], [], [], '2d'
    

# Callback to set the active crystal and configuration
@app.callback(
    Output('message-box', 'value', allow_duplicate=True),
    [Input('update-crystal-button', 'n_clicks')],
    [State('message-box', 'value'),
     State('crystal-id-input', 'value'),
     State('crystal-type-dropdown', 'value'),
     State('lattice-type-dropdown', 'value'),
     State('radius-input', 'value'),
     State('height-slab-input', 'value'),
     State('height-supercell-input', 'value'),

     State('advanced-material-toggle', 'on'),
     State('epsilon-bulk-input', 'value'),
     State('epsilon-atom-input', 'value'),
     State('epsilon-background-input', 'value'),
     State('epsilon-diag-input', 'value'),
     State('epsilon-offdiag-input', 'value'),
     State('E-chi2-diag-input', 'value'),
     State('E-chi2-offdiag-input', 'value'),
     State('E-chi3-diag-input', 'value'),
     State('E-chi3-offdiag-input', 'value'),

     State('runner-selector-dropdown', 'value'),
     State('runner-2-selector-dropdown', 'value'),
     State('interpolation-input', 'value'),
     State('resolution-2d-input', 'value'),
     State('resolution-3d-input', 'value'),
     State('periods-for-epsilon-plot-input', 'value'),
     State('periods-for-field-plot-input', 'value'),
     State('num-bands-input', 'value')
     ],
    prevent_initial_call=True
)
def update_crystal(n_clicks, previous_message, crystal_id, crystal_type, lattice_type, radius, height_slab, height_supercell, advanced_material, epsilon_bulk, epsilon_atom, epsilon_background, epsilon_diag, epsilon_offdiag, E_chi2_diag, E_chi2_offdiag, E_chi3_diag, E_chi3_offdiag, runner_1, runner_2, interpolation, resolution_2d, resolution_3d, periods_for_epsilon_plot, periods_for_field_plot, num_bands):
    if n_clicks is None:
        return previous_message

    global crystal_active, configuration_active
    
    if crystal_type == '2d':
        geometry = Crystal2D.basic_geometry(radius_1=radius, eps_atom_1=epsilon_atom, eps_bulk=epsilon_bulk)
        crystal_active = Crystal2D(
            lattice_type=lattice_type,
            num_bands=num_bands,  # Updated to use the provided num_bands
            resolution=resolution_2d,
            interp=interpolation,
            periods=periods_for_epsilon_plot,
            pickle_id=crystal_id,
            geometry=geometry
        )
    elif crystal_type == 'slab':
        geometry = CrystalSlab.basic_geometry(
            radius_1=radius,
            eps_atom_1=epsilon_atom,
            eps_bulk=epsilon_bulk,
            eps_background=epsilon_background,
            height_slab=height_slab,
            height_supercell=height_supercell
        )
        crystal_active = CrystalSlab(
            lattice_type=lattice_type,
            num_bands=num_bands,  # Updated to use the provided num_bands
            resolution=string_to_vector3(resolution_3d),
            interp=interpolation,
            periods=periods_for_epsilon_plot,
            pickle_id=crystal_id,
            geometry=geometry
        )
    else:
        return previous_message + "\nInvalid crystal type selected."

    configuration_active = {
        'crystal_id': crystal_id,
        'crystal_type': crystal_type,
        'lattice_type': lattice_type,
        'radius': radius,
        'height_slab': height_slab,
        'height_supercell': height_supercell,
        'advanced_material': advanced_material,
        'epsilon_bulk': epsilon_bulk,
        'epsilon_atom': epsilon_atom,
        'epsilon_background': epsilon_background,
        'epsilon_diag': epsilon_diag,
        'epsilon_offdiag': epsilon_offdiag,
        'E_chi2_diag': E_chi2_diag,
        'E_chi2_offdiag': E_chi2_offdiag,
        'E_chi3_diag': E_chi3_diag,
        'E_chi3_offdiag': E_chi3_offdiag,
        'runner_1': runner_1,
        'runner_2': runner_2,
        'interpolation': interpolation,
        'num_bands': num_bands,
        'resolution_2d': resolution_2d,
        'resolution_3d': resolution_3d,
        'periods_for_epsilon_plot': periods_for_epsilon_plot,
        'periods_for_field_plot': periods_for_field_plot,
    }

    new_message = f"""
> The active crystal has been updated with the following configuration:
{configuration_active}
Has the simulation been run yet? {crystal_active.has_been_run}
"""
    return previous_message + new_message


# Callback to save the active crystal, configuration and run_flag to a file as a pickle
@app.callback(
    Output('download-crystal', 'data'),
    Output('message-box', 'value', allow_duplicate=True),
    Input('save-crystal-button', 'n_clicks'),
    State('message-box', 'value'),
    prevent_initial_call=True
)
def save_crystal(n_clicks, previous_message):
    if n_clicks is None:
        return dash.no_update, previous_message
    print("saving")
    global crystal_active, configuration_active

    if crystal_active is None or configuration_active is None:
        print("no active crystal or configuration to save")
        return dash.no_update, previous_message + "\nNo active crystal or configuration to save."
    
    data = pickle.dumps({
        'crystal_active': crystal_active,
        'configuration_active': configuration_active
    })
    b64 = base64.b64encode(data).decode()

    new_message = previous_message + "\nCrystal configuration has been saved successfully."
    print("saved")
    

    return {
        'content': b64,
        'filename': f'crystal_configuration_{configuration_active["crystal_id"]}.pkl',
        'base64': True
        }, new_message

# Callback to load a crystal configuration from a file. Update the crystal-configurator-box and message-box
@app.callback(
    [Output('message-box', 'value', allow_duplicate=True),
        Output('geometry-configurator-box', 'children', allow_duplicate=True),
        Output('material-configurator-box', 'children', allow_duplicate=True),
        Output('solver-configurator-box', 'children', allow_duplicate=True)],
    Input('upload-crystal', 'contents'),
    State('message-box', 'value'),
    prevent_initial_call=True
)
def load_crystal(contents, previous_message):
    print("loading")
    if contents is None:
        return previous_message, dash.no_update, dash.no_update, dash.no_update

    global crystal_active, configuration_active

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    data = pickle.loads(decoded)

    crystal_active = data.get('crystal_active')
    configuration_active = data.get('configuration_active')

    # Update the configurator elements with the loaded configuration
    if configuration_active:
        if configuration_active['crystal_type'] == '2d':
            material_configuration_elements['epsilon-background-input'].hide()
            geometry_configuration_elements['height-slab-input'].hide()
            geometry_configuration_elements['height-supercell-input'].hide()
        elif configuration_active['crystal_type'] == 'slab':
            material_configuration_elements['epsilon-background-input'].show()
            geometry_configuration_elements['height-slab-input'].show()
            geometry_configuration_elements['height-supercell-input'].show()
    
    for key in geometry_configuration_elements.keys():
        target_key = geometry_configuration_elements[key].parameter_id
        geometry_configuration_elements[key].change_value(configuration_active[target_key])

    for key in material_configuration_elements.keys():
        target_key = material_configuration_elements[key].parameter_id
        material_configuration_elements[key].change_value(configuration_active[target_key])

    for key in solver_configuration_elements.keys():
        target_key = solver_configuration_elements[key].parameter_id
        solver_configuration_elements[key].change_value(configuration_active[target_key])

    new_message = previous_message + f"\nCrystal configuration has been loaded successfully:\n{configuration_active}."
    print("loaded")
    return new_message, geometry_configuration_elements_list, material_configuration_elements_list, solver_configuration_elements_list

#function to be called when the plot epsilon button is clicked
def plot_epsilon(epsilon_fig):
    global crystal_active, configuration_active

    if crystal_active is None:
        return go.Figure(), "No active crystal to plot epsilon for."

    crystal_active.run_dumb_simulation()  # Run a dummy simulation to get the epsilon array
    crystal_active.extract_data()
    msg  = "> dumb simulation has been run to quickly collect epsilon data"

    if configuration_active["crystal_type"] == '2d':
        epsilon_fig = crystal_active.plot_epsilon_interactive()
    elif configuration_active["crystal_type"] == 'slab':
        epsilon_fig = crystal_active.plot_epsilon_interactive(opacity=0.2, 
                                                              colorscale='matter', 
                                                              override_resolution_with=-1,
                                                              periods=configuration_active["periods_for_epsilon_plot"])
    else:
        return go.Figure(),  msg + "\nInvalid crystal type selected."
    
    epsilon_fig.update_layout(width=700, height=700)
    print(crystal_active)
    return epsilon_fig, msg + "\nDielectric function plotted"

# Method to run the simulation
def run_simulation(crystal):
    
    if crystal is None:
        return go.Figure(), go.Figure(), "Please update the active crystal before running the simulation."
    
    if crystal.has_been_run is False:
        crystal.set_solver()
        crystal.run_simulation(runner_1 = configuration_active["runner_1"])   #tm
        crystal.run_simulation(runner_1 = configuration_active["runner_2"])   #te
        crystal.extract_data()
        crystal.has_been_run = True
    
    
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown', 'black', 'white']
    bands_fig = go.Figure()
    if isinstance(crystal, Crystal2D):
        epsilon_fig = crystal.plot_epsilon_interactive()
        epsilon_fig.update_layout(width=700, height=700)
        for i, polarization in enumerate(crystal.freqs):
            color = colors[i % len(colors)]
            bands_fig = crystal.plot_bands_interactive(polarization=polarization, color=color, fig=bands_fig)
        bands_fig.update_layout(width=700, height=700)
    elif isinstance(crystal, CrystalSlab):
        epsilon_fig = crystal_active.plot_epsilon_interactive(opacity=0.2, 
                                                              colorscale='matter', 
                                                              override_resolution_with=-1,
                                                              periods=configuration_active["periods_for_epsilon_plot"])
        epsilon_fig.update_layout(width=700, height=700)
        for i, polarization in enumerate(crystal.freqs):
            color = colors[i % len(colors)]
            bands_fig = crystal.plot_bands_interactive(polarization=polarization, color=color, fig=bands_fig)
        bands_fig.update_layout(width=700, height=700)
    else:
        empty_fig = go.Figure().update_layout(title="Invalid crystal type selected.", width=700, height=700)
        return empty_fig, empty_fig
    
    return epsilon_fig, bands_fig, "> Simulation runned.\n Epsilon and bands plotted."

# Callback to show the dielectric function when the button is clicked
@app.callback(
    [Output('epsilon-graph', 'figure', allow_duplicate=True),
     Output('message-box', 'value', allow_duplicate=True)],
    Input('show-dielectric-button', 'n_clicks'),
    State('epsilon-graph', 'figure'),
    State('message-box', 'value'),
    prevent_initial_call=True
)
def show_dielectric(n_clicks, epsilon_fig, previous_message):
    if n_clicks is None:
        return dash.no_update, previous_message + "\nShow Dielectric button not clicked."

    epsilon_fig, msg = plot_epsilon(epsilon_fig)
    return epsilon_fig, previous_message + "\n" + msg

# Callback to run the simulation when the button is clicked. It will plot epsilon and bands
@app.callback(
    [Output('epsilon-graph', 'figure', allow_duplicate=True),
     Output('bands-graph', 'figure', allow_duplicate=True),
     Output('message-box', 'value', allow_duplicate=True)],
    Input('run-simulation-button', 'n_clicks'),
    State('epsilon-graph', 'figure'),
    State('bands-graph', 'figure'),
    State('message-box', 'value'),
    prevent_initial_call=True
)
def run_simulation_callback(n_clicks, epsilon_fig, bands_fig, previous_message):
    global crystal_active

    if n_clicks is None:
        return dash.no_update, dash.no_update, previous_message + "\nRun Simulation button not clicked."
    epsilon_fig, bands_fig, msg = run_simulation(crystal_active)
    return epsilon_fig, bands_fig, previous_message + "\n" + msg + f"\nHas the simulation been run yet? {crystal_active.has_been_run}."

# Callback to update the field plots when the bands plot is clicked
@app.callback(
    [Output('te-field-graph', 'figure', allow_duplicate=True),
     Output('tm-field-graph', 'figure', allow_duplicate=True),
     Output('message-box', 'value', allow_duplicate=True)],
    Input('bands-graph', 'clickData'),
    State('te-field-graph', 'figure'),
    State('tm-field-graph', 'figure'),
    State('message-box', 'value'),
    prevent_initial_call=True
)
def update_field_plots(clickData, te_field_fig, tm_field_fig, previous_message):
    if clickData is None:
        return te_field_fig, tm_field_fig, previous_message + "\nNo point selected in the bands plot."

    global crystal_active

    if crystal_active is None:
        return te_field_fig, tm_field_fig, previous_message + "\nNo active crystal for field plotting."

    if  crystal_active.has_been_run is False:
        return te_field_fig, tm_field_fig, previous_message + "\nSimulation not yet run. Please run the simulation first."

    # Extract the selected k-point data from the clicked bands plot
    kx, ky, kz = clickData['points'][0]['customdata']
    k_point = mp.Vector3(kx, ky, kz)

    # Generate the TE and TM field plots
    print(f"calculating fields at k-point: {kx:.3f}, {ky:.3f}, {kz:.3f}")
    runner_titles = {
        'run_te': 'TE Field',
        'run_tm': 'TM Field',
        'run_zeven': 'TE-like Field',
        'run_zodd': 'TM-like Field'
    }

    te_title = runner_titles[configuration_active["runner_1"]]
    tm_title = runner_titles[configuration_active["runner_2"]]
    te_field_fig = crystal_active.plot_field_interactive(runner=configuration_active["runner_1"], 
                                                         k_point=k_point,  
                                                         title=te_title,
                                                         periods = configuration_active["periods_for_field_plot"])
    tm_field_fig = crystal_active.plot_field_interactive(runner=configuration_active["runner_2"], 
                                                         k_point=k_point,  
                                                         title=tm_title,
                                                         periods = configuration_active["periods_for_field_plot"])

    new_message = previous_message + f"\nFields plotted for k-point ({kx:.3f}, {ky:.3f}, {kz:.3f})."
    
    return te_field_fig, tm_field_fig, new_message


# Callback to toggle the visibility of advanced material configuration
@app.callback(
    Output('material-configurator-box', 'children', allow_duplicate=True),
    Input('advanced-material-toggle', 'on'),
    prevent_initial_call=True
)
def toggle_advanced_configuration(is_advanced):
    if is_advanced:
        material_configuration_elements['epsilon-diag-input'].show()
        material_configuration_elements['epsilon-offdiag-input'].show()
        material_configuration_elements['epsilon-bulk-input'].hide()  
        material_configuration_elements['E-chi2-diag-input'].show()
        material_configuration_elements['E-chi2-offdiag-input'].show()
        material_configuration_elements['E-chi3-diag-input'].show()
        material_configuration_elements['E-chi3-offdiag-input'].show()
    else:
        material_configuration_elements['epsilon-diag-input'].hide()
        material_configuration_elements['epsilon-offdiag-input'].hide()
        material_configuration_elements['epsilon-bulk-input'].show()
        material_configuration_elements['E-chi2-diag-input'].hide()
        material_configuration_elements['E-chi2-offdiag-input'].hide()
        material_configuration_elements['E-chi3-diag-input'].hide()
        material_configuration_elements['E-chi3-offdiag-input'].hide()

    return material_configuration_elements_list

if __name__ == '__main__':
    #%%
    app.run(debug=True)