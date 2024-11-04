#%%
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import dash_daq as daq

class UI_element:
    def __init__(self, id, element=None, visible = True, parameter_id=None):
        self.element = element
        self.id = id
        self.visible = visible
        self.standard_style = {'padding': '10px', "display": "flex"}
        self.container_id = self.id + '_parent'
        self.parameter_id = parameter_id
        
        

    

    def show(self):
        self.visible = True
        self.element.style['display'] = self.standard_style['display']
    
    def hide(self):
        self.visible = False
        self.element.style['display'] = 'none'

    def toggle(self):
        if self.visible:
            self.hide()
        else:
            self.show()

    def change_value(self, new_value):
        self.element.children[1].children.value = new_value

    
    
def dict_to_elements_list(dictionary):
    return  [dictionary[k].element for k in dictionary.keys()]

    
default_values = {
    'crystal_type': '2d',
    'crystal_id': 'crystal_1',
    'lattice_type': 'square',
    'radius': 0.35,
    'epsilon_bulk': 12,
    'epsilon_atom': 1,
    'epsilon_background': 1,
    'height_slab': 0.5,
    'height_supercell': 4,
    'runner_1': 'run_te',
    'runner_2': 'run_tm',
    'interpolation': 5,
    'num_bands': 6,
    'resolution_2d': 32,
    'resolution_3d': (32, 32, 16),
    'epsilon_diag': (4.569, 4.889, 4.889),
    'epsilon_offdiag': (0, 0, 0),
    'E_chi2_diag': (0, 0, 0),
    'E_chi3_diag': (0, 0, 0),
    'periods_for_epsilon_plot': 3,
    'periods_for_field_plot': 5,
    'k_point_max': 0.01,
}

geometry_default_values = {
    'crystal_type': default_values['crystal_type'],
    'crystal_id': default_values['crystal_id'],
    'lattice_type': default_values['lattice_type'],
    'radius': default_values['radius'],
    'height_slab': default_values['height_slab'],
    'height_supercell': default_values['height_supercell'],
}

material_default_values = {
    'epsilon_bulk': default_values['epsilon_bulk'],
    'epsilon_atom': default_values['epsilon_atom'],
    'epsilon_background': default_values['epsilon_background'],
    'epsilon_diag': default_values['epsilon_diag'],
    'epsilon_offdiag': default_values['epsilon_offdiag'],
    'E_chi2_diag': default_values.get('E_chi2_diag', ''),
    'E_chi3_diag': default_values.get('E_chi3_diag', ''),
    
}

solver_default_values = {
    'runner_1': default_values['runner_1'],
    'runner_2': default_values['runner_2'],
    "k_point_max": default_values['k_point_max'],
    'interpolation': default_values['interpolation'],
    'num_bands': default_values['num_bands'],
    'resolution_2d': default_values['resolution_2d'],  # Assuming 2D resolution as default
    'resolution_3d': default_values['resolution_3d'],
    'periods_for_epsilon_plot': default_values['periods_for_epsilon_plot'],
    'periods_for_field_plot': default_values['periods_for_field_plot'],
}









crystal_type_dropdown = UI_element('crystal-type-dropdown', parameter_id='crystal_type')
crystal_type_dropdown.element = dbc.Row(
    [
        dbc.Col(html.Label("Select Photonic Crystal Type"), width=4),
        dbc.Col(dcc.Dropdown(
            id=crystal_type_dropdown.id,
            options=[
                {'label': '2D Photonic Crystal', 'value': '2d'},
                {'label': 'Photonic Crystal Slab', 'value': 'slab'},  # Future option
            ],
            value=default_values["crystal_type"],  # Default to 2D photonic crystal
        ), width=8),
    ],
    style={'padding': '10px', "display": "flex"},
    id=crystal_type_dropdown.container_id
)


crystal_id_input = UI_element('crystal-id-input', parameter_id='crystal_id')
crystal_id_input.element = dbc.Row(
    [
        dbc.Col(html.Label("Crystal ID"), width=4),
        dbc.Col(dcc.Input(id=crystal_id_input.id, type='text', value=default_values['crystal_id'], placeholder='Enter Crystal ID'), width=8),
    ],
    style={'padding': '10px', "display": "flex"},
    id=crystal_id_input.container_id
)

lattice_type_dropdown = UI_element('lattice-type-dropdown', parameter_id='lattice_type')
lattice_type_dropdown.element = dbc.Row(
    [
        dbc.Col(html.Label("Lattice Type"), width=4),
        dbc.Col(dcc.Dropdown(
            id=lattice_type_dropdown.id,
            options=[
                {'label': 'Square', 'value': 'square'},
                {'label': 'Triangular', 'value': 'triangular'}
            ],
            value=default_values['lattice_type'],  # Default to square
        ), width=8),
    ],
    style={'padding': '10px', "display": "flex"},
    id=lattice_type_dropdown.container_id
)

radius_input = UI_element('radius-input', parameter_id='radius')
radius_input.element = dbc.Row(
    [
        dbc.Col(html.Label("Radius"), width=4),
        dbc.Col(dcc.Input(id=radius_input.id, type='number', value=default_values['radius'], step=0.01), width=8),
    ],
    style={'padding': '10px', "display": "flex"},
    id=radius_input.container_id
)

height_slab_input = UI_element('height-slab-input', parameter_id='height_slab')
height_slab_input.element = dbc.Row(
    [
        dbc.Col(html.Label("Height (Slab)"), width=4),
        dbc.Col(dcc.Input(id=height_slab_input.id, type='number', value=default_values['height_slab'], step=0.01), width=8),
    ],
    style={'padding': '10px', "display": "flex"},
    id=height_slab_input.container_id
)

height_supercell_input = UI_element('height-supercell-input', parameter_id='height_supercell')
height_supercell_input.element = dbc.Row(
    [
        dbc.Col(html.Label("Height (Supercell)"), width=4),
        dbc.Col(dcc.Input(id=height_supercell_input.id, type='number', value=default_values['height_supercell'], step=0.1), width=8),
    ],
    style={'padding': '10px', "display": "flex"},
    id=height_supercell_input.container_id
)


geometry_configuration_elements = {}
geometry_configuration_elements = {
    crystal_type_dropdown.id: crystal_type_dropdown,
    crystal_id_input.id: crystal_id_input,
    lattice_type_dropdown.id: lattice_type_dropdown,
    radius_input.id: radius_input,
    height_slab_input.id: height_slab_input,
    height_supercell_input.id: height_supercell_input,
}

geometry_configuration_elements_list = dict_to_elements_list(geometry_configuration_elements)

#_______________________________________________________________
#  Material configuration elements
#_______________________________________________________________

epsilon_bulk_input = UI_element('epsilon-bulk-input', parameter_id='epsilon_bulk')
epsilon_bulk_input.element = dbc.Row(
    [
        dbc.Col(html.Label("Epsilon (Bulk Material)"), width=4),
        dbc.Col(dcc.Input(id=epsilon_bulk_input.id, type='number', value=default_values['epsilon_bulk'], step=0.1), width=8),
    ],
    style={'padding': '10px', "display": "flex"},
    id=epsilon_bulk_input.container_id
)

epsilon_atom_input = UI_element('epsilon-atom-input', parameter_id='epsilon_atom')
epsilon_atom_input.element = dbc.Row(
    [
        dbc.Col(html.Label("Epsilon (Atom)"), width=4),
        dbc.Col(dcc.Input(id=epsilon_atom_input.id, type='number', value=default_values['epsilon_atom'], step=0.1), width=8),
    ],
    style={'padding': '10px', "display": "flex"},
    id=epsilon_atom_input.container_id
)

epsilon_background_input = UI_element('epsilon-background-input', parameter_id='epsilon_background')
epsilon_background_input.element = dbc.Row(
    [
        dbc.Col(html.Label("Epsilon (Background)"), width=4),
        dbc.Col(dcc.Input(id=epsilon_background_input.id, type='number', value=default_values['epsilon_background'], step=0.1), width=8),
    ],
    style={'padding': '10px', "display": "flex"},
    id=epsilon_background_input.container_id
)



epsilon_diag_input = UI_element('epsilon-diag-input', parameter_id='epsilon_diag')
epsilon_diag_input.element = dbc.Row(
    [
        dbc.Col(html.Label("Epsilon (Diagonal)"), width=4),
        dbc.Col(dcc.Input(id=epsilon_diag_input.id, type='text', value=str(default_values['epsilon_diag']), placeholder='Enter epsilon diagonal as (xx, yy, zz)'), width=8),
    ],
    style={'padding': '10px', "display": "flex"},
    id=epsilon_diag_input.container_id
)

epsilon_offdiag_input = UI_element('epsilon-offdiag-input', parameter_id='epsilon_offdiag')
epsilon_offdiag_input.element = dbc.Row(
    [
        dbc.Col(html.Label("Epsilon (Off-Diagonal)"), width=4),
        dbc.Col(dcc.Input(id=epsilon_offdiag_input.id, type='text', value=str(default_values['epsilon_offdiag']), placeholder='Enter epsilon off-diagonal as (xy, yz, zx)'), width=8),
    ],
    style={'padding': '10px', "display": "flex"},
    id=epsilon_offdiag_input.container_id
)


E_chi2_diag_input = UI_element('E-chi2-diag-input', parameter_id='E_chi2_diag')
E_chi2_diag_input.element = dbc.Row(
    [
        dbc.Col(html.Label("E (Chi2 Diagonal)"), width=4),
        dbc.Col(dcc.Input(id=E_chi2_diag_input.id, type='text', value=str(default_values.get('E_chi2_diag', '')), placeholder='Enter E chi2 diagonal as (xx, yy, zz)'), width=8),
    ],
    style={'padding': '10px', "display": "flex"},
    id=E_chi2_diag_input.container_id
)




E_chi3_diag_input = UI_element('E-chi3-diag-input', parameter_id='E_chi3_diag')
E_chi3_diag_input.element = dbc.Row(
    [
        dbc.Col(html.Label("E (Chi3 Diagonal)"), width=4),
        dbc.Col(dcc.Input(id=E_chi3_diag_input.id, type='text', value=str(default_values.get('E_chi3_diag', '')), placeholder='Enter E chi3 diagonal as (xx, yy, zz)'), width=8),
    ],
    style={'padding': '10px', "display": "flex"},
    id=E_chi3_diag_input.container_id
)




material_configuration_elements = {
    epsilon_atom_input.id: epsilon_atom_input,
    epsilon_background_input.id: epsilon_background_input,
    epsilon_bulk_input.id: epsilon_bulk_input, 
    epsilon_diag_input.id: epsilon_diag_input,
    epsilon_offdiag_input.id: epsilon_offdiag_input,
    E_chi2_diag_input.id: E_chi2_diag_input,
    E_chi3_diag_input.id: E_chi3_diag_input,
    
}

epsilon_offdiag_input.hide()
epsilon_diag_input.hide()   
E_chi2_diag_input.hide()

E_chi3_diag_input.hide()




material_configuration_elements_list = dict_to_elements_list(material_configuration_elements)


#_______________________________________________________________
#  Solver configuration elements
#_______________________________________________________________

runner_1_selector_dropdown = UI_element('runner-selector-dropdown', parameter_id='runner_1')
runner_1_selector_dropdown.element = dbc.Row(
    [
        dbc.Col(html.Label("Select Runner"), width=4),
        dbc.Col(dcc.Dropdown(
            id=runner_1_selector_dropdown.id,
            options=[
                {'label': 'TE', 'value': 'run_te'},
                {'label': 'TM', 'value': 'run_tm'},
                {'label': 'TE-like', 'value': 'run_zeven'},
                {'label': 'TM-like', 'value': 'run_zodd'},
            ],
            value=default_values['runner_1'],  # Default to run_te
        ), width=4),
    ],
    style={'padding': '10px', "display": "flex"},
    id=runner_1_selector_dropdown.container_id
)

runner_2_selector_dropdown = UI_element('runner-2-selector-dropdown', parameter_id='runner_2')
runner_2_selector_dropdown.element = dbc.Row(
    [
        dbc.Col(html.Label("Select Runner 2"), width=4),
        dbc.Col(dcc.Dropdown(
            id=runner_2_selector_dropdown.id,
            options=[
                {'label': 'TE', 'value': 'run_te'},
                {'label': 'TM', 'value': 'run_tm'},
                {'label': 'TE-like', 'value': 'run_zeven'},
                {'label': 'TM-like', 'value': 'run_zodd'},
            ],
            value=default_values['runner_2'],  # Default to run_tm
        ), width=4),
    ],
    style={'padding': '10px', "display": "flex"},
    id=runner_2_selector_dropdown.container_id
)

k_point_max_input = UI_element('k-point-max-input', parameter_id='k_point_max')
k_point_max_input.element = dbc.Row(
    [
        dbc.Col(html.Label("K-Point Max"), width=4),
        dbc.Col(dcc.Input(id=k_point_max_input.id, type='number', value=default_values['k_point_max'], step=0.001), width=8),
    ],
    style={'padding': '10px', "display": "flex"},
    id=k_point_max_input.container_id
)




interpolation_input = UI_element('interpolation-input', parameter_id='interpolation')
interpolation_input.element = dbc.Row(
    [
        dbc.Col(html.Label("Interpolation"), width=4),
        dbc.Col(dcc.Input(id=interpolation_input.id, type='number', value=default_values['interpolation'], step=0.1), width=8),
    ],
    style={'padding': '10px', "display": "flex"},
    id=interpolation_input.container_id
)



num_bands_input = UI_element('num-bands-input', parameter_id='num_bands')
num_bands_input.element = dbc.Row(
    [
        dbc.Col(html.Label("Number of Bands"), width=4),
        dbc.Col(dcc.Input(id=num_bands_input.id, type='number', value=default_values['num_bands'], step=1), width=8),
    ],
    style={'padding': '10px', "display": "flex"},
    id=num_bands_input.container_id
)

periods_for_epsilon_plot_input = UI_element('periods-for-epsilon-plot-input', parameter_id='periods_for_epsilon_plot')
periods_for_epsilon_plot_input.element = dbc.Row(
    [
        dbc.Col(html.Label("Periods for Epsilon Plot"), width=4),
        dbc.Col(dcc.Input(id=periods_for_epsilon_plot_input.id, type='number', value=default_values.get('periods_for_epsilon_plot', 1), step=1), width=8),
    ],
    style={'padding': '10px', "display": "flex"},
    id=periods_for_epsilon_plot_input.container_id
)

periods_for_field_plot_input = UI_element('periods-for-field-plot-input', parameter_id='periods_for_field_plot')
periods_for_field_plot_input.element = dbc.Row(
    [
        dbc.Col(html.Label("Periods for Field Plot"), width=4),
        dbc.Col(dcc.Input(id=periods_for_field_plot_input.id, type='number', value=default_values.get('periods_for_field_plot', 1), step=1), width=8),
    ],
    style={'padding': '10px', "display": "flex"},
    id=periods_for_field_plot_input.container_id
)



resolution_2d_input = UI_element('resolution-2d-input', parameter_id='resolution_2d')
resolution_2d_input.element = dbc.Row(
    [
        dbc.Col(html.Label("Resolution (2D)"), width=4),
        dbc.Col(dcc.Input(id=resolution_2d_input.id, type='number', value=default_values['resolution_2d'], step=1), width=8),
    ],
    style={'padding': '10px', "display": "flex"},
    id=resolution_2d_input.container_id
)


resolution_3d_input = UI_element('resolution-3d-input', parameter_id='resolution_3d')
resolution_3d_input.element = dbc.Row(
    [
        dbc.Col(html.Label("Resolution (3D)"), width=4),
        dbc.Col(dcc.Input(id=resolution_3d_input.id, type='text', value=str(default_values['resolution_3d']), placeholder='Enter 3D resolution as (x, y, z)'), width=8),
    ],
    style={'padding': '10px', "display": "flex"},
    id=resolution_3d_input.container_id
)



solver_configuration_elements = {
    runner_1_selector_dropdown.id: runner_1_selector_dropdown,
    runner_2_selector_dropdown.id: runner_2_selector_dropdown,
    interpolation_input.id: interpolation_input,
    num_bands_input.id: num_bands_input,
    periods_for_epsilon_plot_input.id: periods_for_epsilon_plot_input,
    periods_for_field_plot_input.id: periods_for_field_plot_input,
    resolution_2d_input.id: resolution_2d_input,
    resolution_3d_input.id: resolution_3d_input,
    k_point_max_input.id: k_point_max_input,
}

solver_configuration_elements_list = dict_to_elements_list(solver_configuration_elements)



#%%
if __name__ == '__main__':
    #%%
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    toggle_radius_button = dbc.Button("Toggle Radius Visibility", id='toggle-radius-button', color='primary', className='mr-2')
    change_radius_button = dbc.Button("Change Radius Value", id='change-radius-button', color='secondary', className='mr-2')

    app.layout = dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [html.H4("Geometry Configuration")] + 
                                [geometry_configuration_elements[k].element for k in geometry_configuration_elements.keys()]
                            ),
                            style={"border": "1px solid black", "height": "100%"}
                        ),
                        width=4
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [html.H4("Material Configuration")] + 
                                [material_configuration_elements[k].element for k in material_configuration_elements.keys()]
                            ),
                            style={"border": "1px solid black", "height": "100%"}
                        ),
                        width=4
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [html.H4("Solver Configuration")] + 
                                [solver_configuration_elements[k].element for k in solver_configuration_elements.keys()]
                            ),
                            style={"border": "1px solid black", "height": "100%"}
                        ),
                        width=4
                    ),
                ],
                style={"margin-top": "20px"}
            ),
            dbc.Row(
                [toggle_radius_button, change_radius_button],
                style={"margin-top": "20px"}
            )
        ],
        fluid=True,
    )

    @app.callback(
        Output(radius_input.container_id, 'style'),
        [Input('toggle-radius-button', 'n_clicks')],
        [State(radius_input.container_id, 'style')]
    )
    def toggle_radius_visibility(n_clicks, current_style):
        if n_clicks is None:
            raise dash.exceptions.PreventUpdate
        if current_style['display'] == 'none':
            geometry_configuration_elements[radius_input.id].show()
        else:
            geometry_configuration_elements[radius_input.id].hide()
        return geometry_configuration_elements[radius_input.id].element.style

    @app.callback(
        Output(radius_input.id, 'value'),
        [Input('change-radius-button', 'n_clicks')],
        [State(radius_input.id, 'value')]
    )
    def change_radius_value(n_clicks, current_value):
        if n_clicks is None:
            raise dash.exceptions.PreventUpdate
        new_value = current_value + 0.1 if current_value is not None else 0.1
        geometry_configuration_elements[radius_input.id].change_value(new_value)
        return new_value

    app.run_server(debug=True)



# %%
