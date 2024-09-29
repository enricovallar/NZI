#%%
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

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

    
default_values = {
    'crystal_type': '2d',
    'crystal_id': 'crystal_1',
    'lattice_type': 'square',
    'radius': 0.35,
    'epsilon_bulk': 12,
    'epsilon_atom': 1,
    'epsilon_background': 1,
    'height_slab': 0.5,
    'height_supercell': 4
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
        ), width=4),
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
        ), width=4),
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


configuration_elements = {}
configuration_elements = {
    crystal_type_dropdown.id: crystal_type_dropdown,
    crystal_id_input.id: crystal_id_input,
    lattice_type_dropdown.id: lattice_type_dropdown,
    radius_input.id: radius_input,
    epsilon_bulk_input.id: epsilon_bulk_input,
    epsilon_atom_input.id: epsilon_atom_input,
    epsilon_background_input.id: epsilon_background_input,
    height_slab_input.id: height_slab_input,
    height_supercell_input.id: height_supercell_input,
}


def dict_to_elements_list(dictionary):
    return  [dictionary[k].element for k in dictionary.keys()]

# Example usage
configuration_elements_list = dict_to_elements_list(configuration_elements)




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
            value='run_te',  # Default to run_te
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
            value='run_tm',  # Default to run_te
        ), width=4),
    ],
    style={'padding': '10px', "display": "flex"},
    id=runner_2_selector_dropdown.container_id
)

interpolation_input = UI_element('interpolation-input', parameter_id='interpolation')
interpolation_input.element = dbc.Row(
    [
        dbc.Col(html.Label("Interpolation"), width=4),
        dbc.Col(dcc.Input(id=interpolation_input.id, type='number', value=4, step=0.1), width=8),
    ],
    style={'padding': '10px', "display": "flex"},
    id=interpolation_input.container_id
)

runner_configuration_elements = {
    runner_1_selector_dropdown.id: runner_1_selector_dropdown,
    runner_2_selector_dropdown.id: runner_2_selector_dropdown,
    interpolation_input.id: interpolation_input,
}

runner_configuration_elements_list = dict_to_elements_list(runner_configuration_elements)







#%%
if __name__ == '__main__':
    #%%
    
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    toggle_radius_button = dbc.Button("Toggle Radius Visibility", id='toggle-radius-button', color='primary', className='mr-2')
    change_radius_button = dbc.Button("Change Radius Value", id='change-radius-button', color='secondary', className='mr-2')

    app.layout = dbc.Container(
        [toggle_radius_button, change_radius_button] + [configuration_elements[k].element for k in configuration_elements.keys()],
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
            configuration_elements[radius_input.id].show()
        else:
            configuration_elements[radius_input.id].hide()
        return configuration_elements[radius_input.id].element.style

    @app.callback(
        Output(radius_input.id, 'value'),
        [Input('change-radius-button', 'n_clicks')],
        [State(radius_input.id, 'value')]
    )
    def change_radius_value(n_clicks, current_value):
        if n_clicks is None:
            raise dash.exceptions.PreventUpdate
        new_value = current_value + 0.1 if current_value is not None else 0.1
        configuration_elements[radius_input.id].change_value(new_value)
        return new_value
    

    app.layout = dbc.Container(
        [toggle_radius_button, change_radius_button] + 
        [configuration_elements[k].element for k in configuration_elements.keys()] + 
        [runner_configuration_elements[k].element for k in runner_configuration_elements.keys()],
        fluid=True,
    )

    app.run_server(debug=True)



# %%
