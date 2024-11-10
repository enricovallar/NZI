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
from photonic_crystal import Crystal2D, CrystalSlab  
import dash_daq as daq
from crystal_materials import Crystal_Materials
from crystal_geometries import Crystal_Geometry, CrystalSlab_Geometry, Crystal2D_Geometry
import numpy as np
from ui_elements import *




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
active_mode_groups = None
mode_data_to_plot = None


# Create the layout
app.layout = dbc.Container([
    # Add a hidden div to your layout
    html.Div(id='dummy-output', style={'display': 'none'}),
    
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

    
    # Buttons to show the dielectric and run the simulation
    dbc.Row([
        dbc.Col(dbc.Button("Update Crystal", id="update-crystal-button", color="primary"), width=4),
        dbc.Col(dbc.Button("Show Dielectric", id="show-dielectric-button", color="primary"), width=4),
        dbc.Col(dbc.Button("Run Simulation", id="run-simulation-button", color="primary"), width=4),
    ], className="mt-3"),

    # Buttons to save and load the crystal
    dbc.Row([
        dbc.Col(dbc.Button("Save Crystal", id="save-crystal-button", color="secondary"), width={"size": 4}),
        dbc.Col(dcc.Upload(id='upload-crystal', children=dbc.Button("Load Crystal", color="secondary")), width={"size": 4}),
        dcc.Download(id="download-crystal")
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

    # add horizontal line
    html.Hr(style={'borderWidth': '3px'}),

    # Explanation of the field plots
    dbc.Row([
        dbc.Col(html.H4("Select the options for the plots. Click in one point of the band diagram to see the corresponding E and H fields"), width=12),
    ], className="mt-3"),

    # Dropdown menu to choose the mathematical operation (real, imag, abs) to use in the plot functions
    dbc.Row([
        dbc.Col(html.Label("Select Operation"), width=4),
        dbc.Col(dcc.Dropdown(
            id='operation-dropdown',
            options=[
                {'label': 'Real', 'value': 'real'},
                {'label': 'Imaginary', 'value': 'imag'},
                {'label': 'Absolute Value', 'value': 'abs'}
            ],
            value='real'
        ), width=8),
    ], className="mt-3"),

    # Input field for the frequency tolerance
    dbc.Row([
        dbc.Col(html.Label("Frequency Tolerance"), width=4),
        dbc.Col(dcc.Input(
            id='frequency-tolerance-input',
            type='number',
            value=0.01,
            step=0.001,
            min=0,
            max=1,
            style={'width': '100%'}
        ), width=8),
    ], className="mt-3"),

    # Input field for the integer number of periods to plot
    dbc.Row([
        dbc.Col(html.Label("Number of Periods to Plot"), width=4),
        dbc.Col(dcc.Input(
            id='field-periods-to-plot-input',
            type='number',
            value=1,
            step=1,
            min=1,
            style={'width': '100%'}
        ), width=8),
    ], className="mt-3"),

    # Toggle to choose if plot fields with bloch phase or not
    dbc.Row([
        dbc.Col([
            html.Label("Plot Fields with Bloch Phase"),
            daq.BooleanSwitch(id='bloch-phase-toggle', on=True, style={'marginLeft': '10px'})
        ], width=6, style={'display': 'flex', 'alignItems': 'center'}),
    ], className="mt-3"),


    # Button to update the field plots
    dbc.Row([
        dbc.Col(dbc.Button("Update Field Plots", id="update-field-plots-button", color="primary"), width=4),
    ], className="mt-3"),
    
   
    # Placeholder for the electric field plot, three components
    dbc.Row([
        dbc.Col(dcc.Graph(id='e-field-graph', style={'height': '700px', 'width': '1400px'}), width=12),
    ], className="mt-4"),

    # Placeholder for the magnetic field plot, three components
    dbc.Row([
        dbc.Col(dcc.Graph(id='h-field-graph', style={'height': '700px', 'width': '1400px'}), width=12),
    ], className="mt-4"),

    # Bold horizontal line
    html.Hr(style={'borderWidth': '3px'}),
    

    # Box to sweep geometry parameters
    dbc.Card(
        dbc.CardBody([
            html.H4("Geometry Sweep"),
            html.Div(id='geometry-sweep-box', children=[
                dbc.Row(
                    sweep_configuration_elements_list,
                    className="mt-4"),
            ])
        ]),
        style={'border': '2px solid black', 'padding': '10px', 'height': '100%', 'overflowY': 'scroll'}
    ),

    # Placeholder figure to plot the sweep result: 
    dbc.Row([
        dbc.Col(
            dcc.Graph(
                id='sweep-result-graph',
                style={'height': '700px', 'width': '1400px'}
            ),
            width=12
        )
    ], className="mt-4"),
    
])

# callback to update the configurator-box content when a different photonic crystal type is selected
@app.callback(
    [Output('geometry-configurator-box', 'children'),
     Output('material-configurator-box', 'children'),
     Output('solver-configurator-box', 'children'),
    ],
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

    elif crystal_type == 'slab':
        material_configuration_elements['epsilon-background-input'].show()
        geometry_configuration_elements['height-slab-input'].show()
        geometry_configuration_elements['height-supercell-input'].show()
        solver_configuration_elements['resolution-2d-input'].hide()
        solver_configuration_elements['resolution-3d-input'].show()

    geometry_configuration_elements['crystal-type-dropdown'].change_value(crystal_type)  
    return geometry_configuration_elements_list, material_configuration_elements_list, solver_configuration_elements_list
    
    

# Callback to update the geometry configurator box when a different geometry type is selected
@app.callback(
    Output('geometry-configurator-box', 'children', allow_duplicate=True),
    Input('cell-geometry-dropdown', 'value'), 
    State('crystal-type-dropdown', 'value')
)
def update_geometry_configurator(geometry_type, crystal_type):
    global crystal_active, configuration_active
    
    if geometry_type == 'circular':
        geometry_configuration_elements['a-input'].hide()
        geometry_configuration_elements['b-input'].hide()
        geometry_configuration_elements['edge-length-input'].hide()
        geometry_configuration_elements['radius-input'].show()

    elif geometry_type == 'square':
        geometry_configuration_elements['a-input'].hide()
        geometry_configuration_elements['b-input'].hide()
        geometry_configuration_elements['edge-length-input'].show()
        geometry_configuration_elements['radius-input'].hide()
    elif geometry_type == 'elliptical':
        geometry_configuration_elements['a-input'].show()
        geometry_configuration_elements['b-input'].show()
        geometry_configuration_elements['edge-length-input'].hide()
        geometry_configuration_elements['radius-input'].hide()
    elif geometry_type == 'rectangular':
        geometry_configuration_elements['a-input'].show()
        geometry_configuration_elements['b-input'].show()
        geometry_configuration_elements['edge-length-input'].hide()
        geometry_configuration_elements['radius-input'].hide()
    else:
        geometry_configuration_elements['a-input'].hide()
        geometry_configuration_elements['b-input'].hide()
        geometry_configuration_elements['edge-length-input'].hide()
        geometry_configuration_elements['radius-input'].hide()  
    geometry_configuration_elements['cell-geometry-dropdown'].change_value(geometry_type)
    geometry_configuration_elements['crystal-type-dropdown'].change_value(crystal_type)
    return geometry_configuration_elements_list


# Callback to set the active crystal and configuration, reset plots
@app.callback(
    [Output('message-box', 'value', allow_duplicate=True),
    Output('epsilon-graph', 'figure', allow_duplicate=True),
    Output('bands-graph', 'figure', allow_duplicate=True),
    Output('e-field-graph', 'figure', allow_duplicate=True),
    Output('h-field-graph', 'figure', allow_duplicate=True),
    Output('sweep-result-graph', 'figure', allow_duplicate=True),],
    [Input('update-crystal-button', 'n_clicks')],
    [State('message-box', 'value'),
     State('crystal-id-input', 'value'),
     State('crystal-type-dropdown', 'value'),
     State('lattice-type-dropdown', 'value'),
     State('cell-geometry-dropdown', 'value'),
     State('radius-input', 'value'),
     State('a-input', 'value'),
     State('b-input', 'value'),
     State('edge-length-input', 'value'),
     State('height-slab-input', 'value'),
     State('height-supercell-input', 'value'),
     State('advanced-material-toggle', 'on'),
     State('epsilon-bulk-input', 'value'),
     State('epsilon-atom-input', 'value'),
     State('epsilon-background-input', 'value'),
     State('epsilon-diag-input', 'value'),
     State('epsilon-offdiag-input', 'value'),
     State('E-chi2-diag-input', 'value'), 
     State('E-chi3-diag-input', 'value'),
     State('runner-selector-dropdown', 'value'),
     State('runner-2-selector-dropdown', 'value'),
     State('interpolation-input', 'value'),
     State('resolution-2d-input', 'value'),
     State('resolution-3d-input', 'value'),
     State('periods-for-epsilon-plot-input', 'value'),
     State('periods-for-field-plot-input', 'value'),
     State('num-bands-input', 'value'),
     State('k-point-max-input', 'value'),
     ],
    prevent_initial_call=True
)
def update_crystal(n_clicks, previous_message, crystal_id, crystal_type, lattice_type, 
                   geometry_type, radius, a, b, edge_length,
                   height_slab, height_supercell, advanced_material, epsilon_bulk, epsilon_atom, epsilon_background, 
                   epsilon_diag, epsilon_offdiag, E_chi2_diag, E_chi3_diag, runner_1, runner_2, interpolation, 
                   resolution_2d, resolution_3d, periods_for_epsilon_plot, periods_for_field_plot, num_bands, k_point_max):
    if n_clicks is None:
        return previous_message

    global crystal_active, configuration_active, mode_data_to_plot

    mode_data_to_plot = None
    
    if advanced_material is True:
        bulk_material_configuration = {
            'epsilon_diag': epsilon_diag,
            'epsilon_offdiag': epsilon_offdiag,
            'E_chi2_diag': E_chi2_diag,
            'E_chi3_diag': E_chi3_diag
        }
    else:
        bulk_material_configuration = {
            'epsilon': epsilon_bulk
        }
    atom_epsilon = {
        'epsilon': epsilon_atom
    }
    
    material = Crystal_Materials()
    material.background = {"epsilon": 1}
    material.bulk = bulk_material_configuration
    material.atom = atom_epsilon
    material.substrate = {"epsilon": 1}
    if crystal_type == '2d':
        if geometry_type == 'circular':
            radius = float(radius)
            geometry = Crystal2D_Geometry(material=material, geometry_type=geometry_type, r=radius)
        elif geometry_type == 'square':
            edge_length = float(edge_length)
            geometry = Crystal2D_Geometry(material=material, geometry_type=geometry_type, l=edge_length)
        elif geometry_type == 'elliptical':
            a = float(a)
            b = float(b)
            geometry = Crystal2D_Geometry(material=material, geometry_type=geometry_type, a=a, b=b)
        elif geometry_type == "rectangular": 
            a = float(a)
            b = float(b)
            geometry = Crystal2D_Geometry(material=material, geometry_type=geometry_type, a=a, b=b)
        else:
            raise ValueError(f"Invalid geometry type: {geometry_type}")
        
        

        crystal_active = Crystal2D(
            lattice_type=lattice_type,
            geometry=geometry,
            num_bands=num_bands,  # Updated to use the provided num_bands
            resolution=resolution_2d,
            interp=interpolation,
            periods=periods_for_epsilon_plot,
            pickle_id=crystal_id,
            k_point_max=k_point_max
        )
    elif crystal_type == 'slab':
        if geometry_type == 'circular':
            geometry = CrystalSlab_Geometry(material=material, geometry_type=geometry_type, height_slab=height_slab, height_supercell=height_supercell, r=radius)
        elif geometry_type == 'square':
            geometry = CrystalSlab_Geometry(material=material, geometry_type=geometry_type, height_slab=height_slab, height_supercell=height_supercell, a=radius)
        elif geometry_type == 'elliptical':
            geometry = CrystalSlab_Geometry(material=material, geometry_type=geometry_type, height_slab=height_slab, height_supercell=height_supercell, a=radius, b=radius)
        elif geometry_type == "rectangular":
            geometry = CrystalSlab_Geometry(material=material, geometry_type=geometry_type, height_slab=height_slab, height_supercell=height_supercell, a=radius, b=radius)
        else:
            raise ValueError(f"Invalid geometry type: {geometry_type}")
        
        crystal_active = CrystalSlab(
            lattice_type=lattice_type,
            geometry=geometry,
            num_bands=num_bands,  # Updated to use the provided num_bands
            resolution=string_to_vector3(resolution_3d),
            interp=interpolation,
            periods=periods_for_epsilon_plot,
            pickle_id=crystal_id,
            k_point_max=k_point_max,
        )
    else:
        return previous_message + "\nInvalid crystal type selected."

    configuration_active = {
        'crystal_id': crystal_id,
        'crystal_type': crystal_type,
        'lattice_type': lattice_type,
        'geometry_type': geometry_type,
        'radius': radius,
        'a': a,
        'b': b,
        'edge_length': edge_length,
        'height_slab': height_slab,
        'height_supercell': height_supercell,
        'advanced_material': advanced_material,
        'epsilon_bulk': epsilon_bulk,
        'epsilon_atom': epsilon_atom,
        'epsilon_background': epsilon_background,
        'epsilon_diag': epsilon_diag,
        'epsilon_offdiag': epsilon_offdiag,
        'E_chi2_diag': E_chi2_diag,
        'E_chi3_diag': E_chi3_diag,
        'runner_1': runner_1,
        'runner_2': runner_2,
        'interpolation': interpolation,
        'num_bands': num_bands,
        'resolution_2d': resolution_2d,
        'resolution_3d': resolution_3d,
        'periods_for_epsilon_plot': periods_for_epsilon_plot,
        'periods_for_field_plot': periods_for_field_plot,
        'k_point_max': k_point_max,
    }

    new_message = f"""
> The active crystal has been updated to {crystal_active.pickle_id}. Run the simulation to store modes.
  Press Show Dielectric to plot the dielectric function (plotted for the gamma point, 1st band).
"""
    
    fig = go.Figure()
    return previous_message + new_message, fig, fig, fig, fig, fig


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

        if configuration_active['advanced_material'] is True:
            material_configuration_elements['epsilon-diag-input'].show()
            material_configuration_elements['epsilon-offdiag-input'].show()
            material_configuration_elements['epsilon-bulk-input'].hide()
            material_configuration_elements['E-chi2-diag-input'].show()
            material_configuration_elements['E-chi3-diag-input'].show()
        else:
            material_configuration_elements['epsilon-diag-input'].hide()
            material_configuration_elements['epsilon-offdiag-input'].hide()
            material_configuration_elements['epsilon-bulk-input'].show()
            material_configuration_elements['E-chi2-diag-input'].hide()
            material_configuration_elements['E-chi3-diag-input'].hide()
        
        if configuration_active['cell-geometry-dropdown'] == 'circular':
            geometry_configuration_elements['a-input'].hide()
            geometry_configuration_elements['b-input'].hide()
            geometry_configuration_elements['edge-length-input'].hide()
            geometry_configuration_elements['radius-input'].show()
        elif configuration_active['cell-geometry-dropdown'] == 'square':
            geometry_configuration_elements['a-input'].hide()
            geometry_configuration_elements['b-input'].hide()
            geometry_configuration_elements['edge-length-input'].show()
            geometry_configuration_elements['radius-input'].hide()
        elif configuration_active['cell-geometry-dropdown'] == 'elliptical':
            geometry_configuration_elements['a-input'].show()
            geometry_configuration_elements['b-input'].show()
            geometry_configuration_elements['edge-length-input'].hide()
            geometry_configuration_elements['radius-input'].hide()
        elif configuration_active['cell-geometry-dropdown'] == 'rectangular':
            geometry_configuration_elements['a-input'].show()
            geometry_configuration_elements['b-input'].show()
            geometry_configuration_elements['edge-length-input'].hide()
            geometry_configuration_elements['radius-input'].hide()
        else:
            geometry_configuration_elements['a-input'].hide()
            geometry_configuration_elements['b-input'].hide()
            geometry_configuration_elements['edge-length-input'].hide()
            geometry_configuration_elements['radius-input'].hide()
    
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



# function to be called when the plot epsilon button is clicked
def plot_epsilon(epsilon_fig):
    global crystal_active, configuration_active

    if crystal_active is None:
        return go.Figure(), "No active crystal to plot epsilon for."

    crystal_active.run_dumb_simulation()  # Run a dummy simulation to get the epsilon array
    crystal_active.extract_data()
    msg  = "> A dumb simulation has been run to quickly collect epsilon data in gamma point (1st band)."

    if configuration_active["crystal_type"] == '2d':
        epsilon_fig = crystal_active.plot_epsilon()
    elif configuration_active["crystal_type"] == 'slab':
        epsilon_fig = crystal_active.plot_epsilon(opacity=0.2, 
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
        crystal.run_simulation(runner = configuration_active["runner_1"])   #tm
        crystal.run_simulation(runner = configuration_active["runner_2"])   #te
        crystal.extract_data()
        crystal.has_been_run = True
    
    
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown', 'black', 'white']
    bands_fig = go.Figure()
    if isinstance(crystal, Crystal2D):
        epsilon_fig = crystal.plot_epsilon()
        epsilon_fig.update_layout(width=700, height=700)
        for i, polarization in enumerate(crystal.freqs):
            color = colors[i % len(colors)]
            bands_fig = crystal.plot_bands(polarization=polarization, color=color, fig=bands_fig)
        bands_fig.update_layout(width=700, height=700)
    elif isinstance(crystal, CrystalSlab):
        epsilon_fig = crystal_active.plot_epsilon(opacity=0.2, 
                                                              colorscale='matter', 
                                                              override_resolution_with=-1,
                                                              periods=configuration_active["periods_for_epsilon_plot"])
        epsilon_fig.update_layout(width=700, height=700)
        for i, polarization in enumerate(crystal.freqs):
            color = colors[i % len(colors)]
            bands_fig = crystal.plot_bands(polarization=polarization, color=color, fig=bands_fig)
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
# and it will group the modes by k-point
@app.callback(
    [Output('epsilon-graph', 'figure', allow_duplicate=True),
     Output('bands-graph', 'figure', allow_duplicate=True),
     Output('message-box', 'value', allow_duplicate=True),],
    Input('run-simulation-button', 'n_clicks'),
    State('epsilon-graph', 'figure'),
    State('bands-graph', 'figure'),
    State('message-box', 'value'),
    prevent_initial_call=True
)
def run_simulation_callback(n_clicks, epsilon_fig, bands_fig, previous_message):
    global crystal_active

    if crystal_active is None:
        return dash.no_update, dash.no_update, previous_message + "\nNo active crystal to run the simulation."

    if n_clicks is None:
        return dash.no_update, dash.no_update, previous_message + "\nRun Simulation button not clicked."

    epsilon_fig, bands_fig, msg = run_simulation(crystal_active)

    
    return epsilon_fig, bands_fig, previous_message + "\n" + msg + f"\nHas the simulation been run yet? {crystal_active.has_been_run}."

# Callback to update the field plots when the bands plot is clicked
@app.callback(
    [Output('e-field-graph', 'figure', allow_duplicate=True),
     Output('h-field-graph', 'figure', allow_duplicate=True),
     Output('message-box', 'value', allow_duplicate=True)],
    Input('bands-graph', 'clickData'),
    State('e-field-graph', 'figure'),
    State('h-field-graph', 'figure'),
    State('frequency-tolerance-input', 'value'),
    State('operation-dropdown', 'value'),
    State('field-periods-to-plot-input', 'value'),
    State('bloch-phase-toggle', 'on'),
    State('message-box', 'value'),
    prevent_initial_call=True
)
def update_field_plots_by_click(clickData, e_field_fig, h_field_fig, frequency_tolerance, operation, periods, bloch_phase, previous_message):
    if clickData is None:
        return e_field_fig, h_field_fig, previous_message + "\nNo point selected in the bands plot."
    global crystal_active, mode_data_to_plot
    

    if crystal_active is None:
        return e_field_fig, h_field_fig, previous_message + "\nNo active crystal for field plotting."

    if  crystal_active.has_been_run is False:
        return e_field_fig, h_field_fig, previous_message + "\nSimulation not yet run. Please run the simulation first."
    
    
    # Extract the selected k-point data from the clicked bands plot
    kx, ky, kz, freq, polarization = clickData['points'][0]['customdata']
    k_point = mp.Vector3(kx, ky, kz)

    mode_data_to_plot = {
        'k_point': k_point,
        'freq': freq,
        'polarization': polarization,
    }

    e_field_fig, h_field_fig = crystal_active.plot_field_components(polarization, k_point, freq, 
                                                       frequency_tolerance=frequency_tolerance, 
                                                       k_point_max_distance=None, 
                                                       quantity=operation, 
                                                       periods=periods,
                                                       bloch_phase=bloch_phase)

    e_field_fig.update_layout(width=1400, height=700)
    h_field_fig.update_layout(width=1400, height=700)
    new_message = previous_message + f"\nFields plotted for k-point ({kx:.3f}, {ky:.3f}, {kz:.3f}) and frequency {freq:0.4f}."
    
    return e_field_fig, h_field_fig, new_message

# Callback to update the field plots when the update field plots button is clicked
@app.callback(
    [Output('e-field-graph', 'figure', allow_duplicate=True),
     Output('h-field-graph', 'figure', allow_duplicate=True),
     Output('message-box', 'value', allow_duplicate=True)],
    Input('update-field-plots-button', 'n_clicks'),
    State('e-field-graph', 'figure'),
    State('h-field-graph', 'figure'),
    State('frequency-tolerance-input', 'value'),
    State('operation-dropdown', 'value'),
    State('field-periods-to-plot-input', 'value'),
    State('bloch-phase-toggle', 'on'),
    State('message-box', 'value'),
    prevent_initial_call=True
)
def update_field_plots_button(n_clicks, e_field_fig, h_field_fig, frequency_tolerance, operation, periods,bloch_phase, previous_message):
    if n_clicks is None:
        return e_field_fig, h_field_fig, previous_message + "\nUpdate Field Plots button not clicked."
    
    global crystal_active, mode_data_to_plot

    if crystal_active is None:
        return e_field_fig, h_field_fig, previous_message + "\nNo active crystal for field plotting."

    if crystal_active.has_been_run is False:
        return e_field_fig, h_field_fig, previous_message + "\nSimulation not yet run. Please run the simulation first."

    if mode_data_to_plot is None:
        return e_field_fig, h_field_fig, previous_message + "\nNo mode data available for field plotting."

    k_point = mode_data_to_plot['k_point']
    freq = mode_data_to_plot['freq']
    polarization = mode_data_to_plot['polarization']

    e_field_fig, h_field_fig = crystal_active.plot_field_components(polarization, k_point, freq, 
                                                       frequency_tolerance=frequency_tolerance, 
                                                       k_point_max_distance=None, 
                                                       quantity=operation, 
                                                       periods=periods, 
                                                       bloch_phase=bloch_phase)

    e_field_fig.update_layout(width=1400, height=700)
    h_field_fig.update_layout(width=1400, height=700)
    new_message = previous_message + f"\nFields updated for k-point ({k_point.x:.3f}, {k_point.y:.3f}, {k_point.z:.3f}) and frequency {freq:0.4f}."
    
    return e_field_fig, h_field_fig, new_message

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
        material_configuration_elements['E-chi3-diag-input'].show()
    else:
        material_configuration_elements['epsilon-diag-input'].hide()
        material_configuration_elements['epsilon-offdiag-input'].hide()
        material_configuration_elements['epsilon-bulk-input'].show()
        material_configuration_elements['E-chi2-diag-input'].hide()
        material_configuration_elements['E-chi3-diag-input'].hide()

    return material_configuration_elements_list


# Callback to sweep the geometry parameters
@app.callback(
    [Output('sweep-result-graph', 'figure', allow_duplicate=True),
    Output('message-box', 'value', allow_duplicate=True)],
    Input('run-sweep-button', 'n_clicks'),
    State('sweep-parameter-dropdown', 'value'),
    State('sweep-range-start-input', 'value'),
    State('sweep-range-end-input', 'value'),
    State('sweep-steps-input', 'value'),
    State('message-box', 'value'),
    prevent_initial_call=True
)
def run_sweep(n_clicks, sweep_parameter, start, end, steps, previous_message):
    print("running sweep")
    global crystal_active, configuration_active
    if n_clicks is None:
        return dash.no_update

    print("...")
    sweep_values = np.linspace(start, end, steps)
    sweep_results = crystal_active.sweep_geometry_parameter(sweep_parameter, sweep_values, crystal_active.num_bands)
    fig = crystal_active.plot_sweep_result(sweep_results)
    msg = f"Sweep results plotted for parameter {sweep_parameter}.\n"

    return fig, previous_message + msg

    #%%
app.run(debug=True)