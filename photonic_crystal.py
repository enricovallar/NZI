#%%
import math
import meep as mp
from meep import mpb
import pickle
import contextlib
import os
import sys
import plotly.graph_objects as go
import numpy as np
import group_theory_analysis as gta
from plotly.subplots import make_subplots
from collections import defaultdict


class PhotonicCrystal:
    def __init__(self,
                lattice_type = None,
                num_bands: int = 6,
                resolution: tuple[int, int] | int = 32,
                interp: int = 4,
                periods: int = 3, 
                pickle_id = None, 
                k_points = None, 
                ):
        
        self.lattice_type = lattice_type
        self.num_bands = num_bands
        self.resolution = resolution
        self.interp = interp
        self.periods = periods
        self.pickle_id = pickle_id
        self.has_been_run = False #update this manually

        #this values are set with basic lattice method
        self.geometry_lattice= None 
        self.k_points = k_points
        if self.k_points is not None:
            self.k_points_interpolated = mp.interpolate(self.interp, self.k_points)
        
        #slef.geometry_lattice, self.k_points = self.basic_lattice()

        #this values are set with basic geometry method
        self.basic_geometry = None
        #self.geometry = self.basic_geometry()


        self.ms = None
        self.md = None

        self.freqs = {}
        self.gaps = {}
        self.epsilon = None
        self.modes= []

    def __getstate__(self):
        state = self.__dict__.copy()
        # Exclude the non-picklable SWIG objects
        state['ms'] = None
        state['md'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # You may want to reinitialize 'ms' and 'md' if needed after loading.
        self.ms = None
        self.md = None

    def pickle_photonic_crystal(self, pickle_id):
        with open(f"{pickle_id}.pkl", "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_photonic_crystal(pickle_id):
        with open(f"{pickle_id}.pkl", "rb") as f:
            return pickle.load(f)

    def set_solver(self, k_point = None):

        if k_point is not None:
            self.ms = mpb.ModeSolver(geometry=self.geometry,
                                  geometry_lattice=self.geometry_lattice,
                                  k_points=[k_point],
                                  resolution=self.resolution,
                                  num_bands=self.num_bands)
        else:
            self.ms = mpb.ModeSolver(geometry=self.geometry,
                                    geometry_lattice=self.geometry_lattice,
                                    k_points=self.k_points_interpolated,
                                    resolution=self.resolution,
                                    num_bands=self.num_bands)

    def run_simulation(self, runner="run_zeven", polarization=None):
        """
        Run the simulation to calculate the frequencies and gaps.
        
        Parameters:
        - runner: The name of the function to run the simulation. Default is 'run_zeven'.
        """
        if self.ms is None:
            raise ValueError("Solver is not set. Call set_solver() before running the simulation.")
        
        if polarization is not None:
            polarization = polarization
        else:
            if runner.startswith("run_"):
                polarization = runner[4:]
            else:
                polarization = runner
        

        
        
        def get_mode_data(ms, band):
            mode = {
                "h_field": ms.get_hfield(band, bloch_phase=True),
                "e_field": ms.get_efield(band, bloch_phase=True),
                "freq": ms.freqs[band-1],
                "k_point": ms.current_k,
                "polarization": polarization
            }
            self.modes.append(mode)
    

    
        print(self.k_points_interpolated)
        with suppress_output():
            getattr(self.ms, runner)(get_mode_data)
            self.freqs[polarization] = self.ms.all_freqs
            self.gaps[polarization] = self.ms.gap_list
        print(self.modes[1])

    def run_dumb_simulation(self):
        """
        Run a dumb simulation. Mainly used in  
        """
        self.ms = mpb.ModeSolver(geometry=self.geometry,
                                  geometry_lattice=self.geometry_lattice,
                                  k_points=[mp.Vector3()],
                                  resolution=self.resolution,
                                  num_bands=1)
        
        self.ms.run()
                
    def extract_data(self, periods: int | None = 5):
        """
        Extract the data from the simulation.
        
        Parameters:
        - periods: The number of periods to extract. Default is 5.
        """
        if self.ms is None:
            raise ValueError("Solver is not set. Call set_solver() before extracting data.")

        self.md = mpb.MPBData(rectify=True, periods=periods, resolution=self.resolution)
        
    
    def plot_epsilon_interactive(self, fig=None, title='Epsilon'):
        raise NotImplementedError

    def plot_bands_interactive(self, polarization="te", title='Bands', fig=None, color='blue'):
        """
        Plot the band structure of the photonic crystal interactively using Plotly, 
        with k-points displayed on hover, click, and rectangular selection events.
        Supports rectangular selection and toggling visibility by clicking on legend.
        """
        if self.freqs[polarization] is None:
            print("Simulation not run yet. Please run the simulation first.")
            return
        freqs = self.freqs[polarization]
        gaps = self.gaps[polarization]

        xs = list(range(len(freqs)))

        # Extract the interpolated k-points as vectors and format them for hover and click
        k_points_interpolated = [kp for kp in self.k_points_interpolated]

        if fig is None:
            fig = go.Figure()

        # Iterate through each frequency band and add them to the plot
        for band_index, band in enumerate(zip(*freqs)):
            # Generate hover text with the corresponding k-point and frequency
            hover_texts = [
                f"k-point: ({kp.x:.4f}, {kp.y:.4f}, {kp.z:.4f})<br>frequency: {f:.4f}"
                for kp, f in zip(k_points_interpolated, band)
            ]
            # Add the line trace with hover info
            fig.add_trace(go.Scatter(
                x=xs, 
                y=band, 
                mode='lines', 
                line=dict(color=color),
                text=hover_texts,  # Custom hover text
                hoverinfo='text',  # Display only the custom hover text
                customdata=[(kp.x, kp.y, kp.z, f) for kp, f in zip(k_points_interpolated, band)],  # Attach k-points and frequency as custom data
                showlegend=False,  # Hide from legend (we’ll add a separate legend entry)
                legendgroup=polarization,  # Group traces by polarization for toggling visibility
                visible=True,  # Initially visible
                selectedpoints=[],  # Placeholder for selected points
                selected=dict(marker=dict(color="red", size=10)),  # Change color and size of selected points
                unselected=dict(marker=dict(opacity=0.3))  # Make unselected points more transparent
            ))

        # Add bandgap shading (optional, grouped with the polarization for toggling visibility)
        for gap in gaps:
            if gap[0] > 1:
                fig.add_shape(
                    type="rect",
                    x0=xs[0], 
                    x1=xs[-1],
                    y0=gap[1], 
                    y1=gap[2],
                    fillcolor=color, 
                    opacity=0.2, 
                    line_width=0,
                    layer="below",
                    legendgroup=polarization,  # Group shading with the same polarization
                    visible=True  # Initially visible
                )

        # Add a single legend entry for toggling visibility
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(color=color),
            name=f'{polarization.upper()}',  # Legend entry for the polarization
            legendgroup=polarization,  # Group with the same polarization traces
            showlegend=True,  # Show the legend entry
            visible=True,  # Initially visible
        ))

        # Customize the x-axis with the high symmetry points
        k_high_sym = self.get_high_symmetry_points()
        fig.update_layout(
            title=title,
            xaxis=dict(
                tickmode='array',
                tickvals=[i * (len(freqs) - 4) / 3 + i for i in range(4)],
                ticktext=list(k_high_sym.keys()) + [list(k_high_sym.keys())[0]]  # Repeat the first element at the end
            ),
            yaxis_title='frequency (c/a)',
            showlegend=True,
            dragmode='select',  # Enables rectangular selection
            clickmode='event+select',  # Enable click events and selection events
            legend=dict(  # This ensures that clicking the legend will toggle visibility
                itemclick="toggle",  # Toggle visibility when clicked
                itemdoubleclick="toggleothers"  # Double-click to hide/show other entries
            )
        )

        # Add a JavaScript callback to handle select events
        fig.update_layout(
            clickmode='event+select'  # Enable click events
        )

        return fig

    def get_high_symmetry_points(self):
        k_high_sym = {}
        if self.lattice_type == 'square':
            k_high_sym = {
                'Γ': mp.Vector3(0, 0, 0),
                'X': mp.Vector3(0.5, 0, 0),
                'M': mp.Vector3(0.5, 0.5, 0)
            }
        elif self.lattice_type == 'triangular':
            k_high_sym = {
                'Γ': mp.Vector3(0, 0, 0),
                'K': mp.Vector3(1/3, 1/3, 0),
                'M': mp.Vector3(0.5, 0, 0)
            }
        return k_high_sym

    def plot_field_interactive(self, runner="run_tm", k_point=mp.Vector3(1 / -3, 1 / 3), frequency=None, periods=5, resolution=32, fig=None, title="Field Visualization", colorscale='RdBu'):
        raise NotImplementedError

    @staticmethod
    def basic_geometry():
        raise NotImplementedError
    
    @staticmethod
    def basic_lattice():
        raise NotImplementedError
    

    def find_modes_symmetries(self):
        if self.freqs is None:
            raise ValueError("Frequencies are not calculated. Run the simulation first.")
        symmetries = {}
        for mode in self.modes:
            mode["symmetries"] = gta.test_symmetries(np.array(mode["field"]))
        return symmetries
    


    def plot_mode_field_vector(self, mode):
        fields = [mode["e_field"], mode["h_field"]]
        names = ["Electric Field", "Magnetic Field"]
        sizerefs = [1, 1]
        fig = self._plot_field_vector(fields, names, sizerefs)
        return fig
    

    def plot_mode_fields_norm_to_k(self, mode, k):
        fields = [mode["e_field"], mode["h_field"]]
        fields_norm_to_k = self._calculate_field_norm_to_k(fields, k)
        names = [f"Electric Field (Perpendicular to k={k})", f"Magnetic Field (Perpendicular to k={k})"]
        sizerefs = [1, 1]
        fig = self._plot_field_vector(fields_norm_to_k, names, sizerefs)
        return fig
    
    
    

    def group_modes_by_k_point(self):
        """
        Groups modes by their k_point.

        Returns:
            list: A list of groups where each group contains modes with the same k_point.
        """
        # Use defaultdict to automatically create a list for each unique k_point
        k_point_groups = defaultdict(list)

        if not self.modes:
            raise ValueError("Modes are not calculated. Run the simulation first.")

        # Iterate through each mode
        for mode in self.modes:
            # Get the k_point of the current mode (assume mode["k_point"] exists)
            k_point = tuple(mode["k_point"])  # Use tuple since lists are not hashable
            # Append the mode to the corresponding k_point group
            k_point_groups[k_point].append(mode)

        # Return the groups as a list of lists
        return list(k_point_groups.values())

        
    



    def _plot_field_vector(fields, names=["Field 1", "Field 2"], sizerefs =[1,1], fig=None):
        """
        Plots a 3D cone plot of one or two vector fields using Plotly.

        Parameters:
        fields (list of numpy.ndarray): A list containing one or two 4D numpy arrays representing the vector fields. 
                                        The last dimension should have size 3, corresponding to the x, y, and z components of the field.
        names (list of str): A list containing the names of the fields to be used as subplot titles.
        fig (plotly.graph_objs._figure.Figure, optional): An existing Plotly figure to which the cone plot will be added. 
                                                        If None, a new figure will be created.

        Returns:
        None: The function displays the plot using Plotly's `show` method.
        """
        
        # Determine the number of subplots based on the number of fields
        num_fields = len(fields)
        if num_fields == 1:
            rows, cols = 1, 1
        elif num_fields == 2:
            rows, cols = 1, 2
        else:
            raise ValueError("fields list can only contain one or two fields.")
        
        if fig is None:
            # Create subplots if no figure is provided, with titles for each subplot
            fig = make_subplots(rows=rows, cols=cols, specs=[[{'type': 'cone'} for _ in range(cols)]],
                                subplot_titles=names)

        for i, field in enumerate(fields):
            # Extract the real parts of the field components
            field_x = np.real(field[..., 0])
            field_y = np.real(field[..., 1])
            field_z = np.real(field[..., 2])

            # Create a meshgrid for the x, y, z coordinates
            x, y, z = np.meshgrid(np.arange(field.shape[0]), np.arange(field.shape[1]), np.arange(field.shape[2]))

            # Add a cone plot to the figure
            # Adjust colorbar position to be next to the respective subplot
            fig.add_trace(go.Cone(
                x=x.flatten(),
                y=y.flatten(),
                z=z.flatten(),
                u=field_x.flatten(),
                v=field_y.flatten(),
                w=field_z.flatten(),
                sizemode='scaled',
                anchor = 'tail',
                sizeref=sizerefs[i],
                colorbar=dict(
                    x=0.45 if i == 0 else 0.95,  # Position colorbar near the first and second subplot
                    y=0.5,  # Vertically center the colorbar
                    thickness=15,
                    len=0.75  # Adjust the length of the colorbar
                )
            ), row=1, col=i+1)

        # Update layout for subplots, adjust margins
        fig.update_layout(
            title='3D Cone Plot of Vector Fields',
            margin=dict(l=50, r=50, t=50, b=50)  # Adjust subplot margins
        )

        # Show the figure
        fig.show()



    def _calculate_field_norm_to_k(fields, k):
        """
        Calculate the components of the field perpendicular to the wavevector k for each field in the list.

        Args:
        - fields: A list of numpy arrays, each of shape (Nx, Ny, Nz, 3), where the last dimension contains the x, y, z components of the field.
        - k: A numpy array of shape (3,), representing the wavevector [kx, ky, kz].
        
        Returns:
        - fields_norm_to_k: A list of numpy arrays, each of shape (Nx, Ny, Nz, 3), representing the field perpendicular to k for each input field.
        """
        fields_norm_to_k = []
        
        for field in fields:
            # Normalize the wavevector k
            k_norm = k / np.linalg.norm(k)
            
            # Compute the dot product of each field vector with the normalized k
            dot_product = np.einsum('ijkl,l->ijk', field, k_norm)  # Efficiently computes the dot product
            
            # Compute the parallel component of the field: (dot_product * k_norm)
            field_parallel = np.outer(dot_product, k_norm).reshape(field.shape)
            
            # Subtract the parallel component to get the perpendicular component
            field_norm_to_k = field - field_parallel
            
            fields_norm_to_k.append(field_norm_to_k)
        
        return fields_norm_to_k





    


@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


class Crystal2D(PhotonicCrystal):
    def __init__(self,
                lattice_type = None,
                num_bands: int = 6,
                resolution: tuple[int, int] | int = 32,
                interp: int =4,
                periods: int =3, 
                pickle_id = None,
                geometry = None):
        super().__init__(lattice_type, num_bands, resolution, interp, periods, pickle_id)
        
        
        self.geometry_lattice, self.k_points = self.basic_lattice(lattice_type)
        self.geometry = geometry if geometry is not None else self.basic_geometry()
        self.k_points_interpolated = mp.interpolate(interp, self.k_points)
        


    def plot_epsilon_interactive(self, fig=None, title='Epsilon', **kwargs):
        """
        Plot the epsilon values obtained from the simulation interactively using Plotly.
        """
        with suppress_output():
            if self.epsilon is None:
                
                md = mpb.MPBData(rectify=True, periods=self.periods, resolution=self.resolution)
                converted_eps = md.convert(self.ms.get_epsilon())
            else:
                converted_eps = self.epsilon
            if fig is None:
                fig = go.Figure()

            fig.add_trace(go.Heatmap(z=converted_eps, colorscale='Viridis'))
            fig.update_layout(
                title=dict(
                    text=f"{title}<br>Dielectric Distribution",
                    x=0.5,
                    y=0.95,
                    xanchor='center',
                    yanchor='top'
                ),
                coloraxis_colorbar=dict(title='$\\epsilon $'),
                xaxis_showgrid=False, 
                yaxis_showgrid=False,
                xaxis_zeroline=False, 
                yaxis_zeroline=False,
                xaxis_visible=False, 
                yaxis_visible=False
            )       
        self.epsilon = converted_eps
        print(self.epsilon)    
        return fig
        

    
    
    
    def plot_field_interactive(self, 
                               runner="run_tm", 
                               k_point=mp.Vector3(0, 0), 
                               periods=5, 
                               fig=None,
                               title="Field Visualization", 
                               colorscale='RdBu'):
        
        
        
        raw_fields = []
        freqs = []

        self.ms = mpb.ModeSolver(geometry=self.geometry,
                                geometry_lattice=self.geometry_lattice,
                                k_points=[k_point],
                                resolution=self.resolution,
                                num_bands=self.num_bands)
        
        def get_zodd_fields(ms, band):
            raw_fields.append(ms.get_hfield(band, bloch_phase=True))
        def get_zeven_fields(ms, band):
            raw_fields.append(ms.get_efield(band, bloch_phase=True))
        def get_freqs(ms, band):
            freqs.append(ms.freqs[band-1])

        
        with suppress_output():
            if runner == "run_te" or runner == "run_zeven":
                self.ms.run_te(mpb.output_at_kpoint(k_point, mpb.fix_hfield_phase, get_zodd_fields, get_freqs))
                field_type = "H-field"
                print(f"frequencies: {freqs}")
                
            elif runner == "run_tm" or runner == "run_zodd":
                self.ms.run_tm(mpb.output_at_kpoint(k_point, mpb.fix_efield_phase, get_zeven_fields, get_freqs))
                field_type = "E-field"
                print(f"frequencies: {freqs}")
                
            else:
                raise ValueError("Invalid runner. Please enter 'run_te', 'run_zeven', or 'run_tm' or 'run_zodd'.")
            
            md = mpb.MPBData(rectify=True, periods=periods, resolution=self.resolution)
            fields = []        
            for field in raw_fields:
                field = field[..., 0, 2]  # Get just the z component of the fields
                
                fields.append(md.convert(field))

            
            
            eps = md.convert(self.ms.get_epsilon())
        #print(fields)
        num_plots = len(fields)
        if num_plots == 0:
            print("No field data to plot.")
            return

        if fig is None:
            fig = go.Figure()

        # Automatically generate the subtitle with the k-vector and field type
        subtitle = f"{field_type}, z-component<br>k = ({k_point.x:.4f}, {k_point.y:.4f})"

        # Initialize an empty list for dropdown menu options
        dropdown_buttons = []

        # Calculate the midpoint between min and max of the permittivity (eps)
        min_eps, max_eps = np.min(eps), np.max(eps)
        midpoint = (min_eps + max_eps) / 2  # The level to be plotted

        for i, (field, freq) in enumerate(zip(fields, freqs)):
            visible_status = [False] * (2 * num_plots)
            visible_status[2 * i] = True  # Make the current contour (eps) visible
            visible_status[2 * i + 1] = True  # Make the current heatmap (field) visible

            # Add the contour plot for permittivity (eps) at the midpoint
            fig.add_trace(go.Contour(z=eps.T,
                                    contours=dict(
                                        start=midpoint,  # Start and end at the midpoint to ensure a single level
                                        end=midpoint,
                                        size=0.1,  # A small size to keep it as a single contour
                                        coloring='none'  # No filling
                                    ),
                                    line=dict(color='black', width=2),
                                    showscale=False,
                                    opacity=0.7,
                                    visible=True if i == 0 else False))  # Initially visible only for the first plot
            
            # Add the heatmap for the real part of the electric field
            fig.add_trace(go.Heatmap(z=np.real(field).T, colorscale=colorscale, zsmooth='best', opacity=0.9,
                                    showscale=False, visible=True if i == 0 else False))

            # Create a button for each field dataset for the dropdown
            dropdown_buttons.append(dict(label=f'Mode {i + 1}',
                                        method='update',
                                        args=[{'visible': visible_status},  # Update visibility for both eps and field
                                            {'title':f"{title}<br>Mode {i + 1}, freq={freq:0.3f}: {subtitle}"}
                                        ]))
            mode = {
                "k_point" : k_point,
                "frequency" : freq,
                "field" : field,
                "field_type" : field_type,
            }
            

            self.modes.append(mode) 


               

        # Add the dropdown menu to the layout
        fig.update_layout(
            updatemenus=[dict(active=0,  # The first dataset is active by default
                            buttons=dropdown_buttons,
                            x=1.15, y=1.15,  # Positioning the dropdown to the top right
                            xanchor='left', yanchor='top')],
            title=f"{title}<br>Mode {1}, freq={freqs[0]:0.3f}: {subtitle}",  # Main title + subtitle
            xaxis_showgrid=False, yaxis_showgrid=False,
            xaxis_zeroline=False, yaxis_zeroline=False,
            xaxis_visible=False, yaxis_visible=False,
            hovermode="closest",
            width=800, height=800
        )
        self.find_modes_symmetries()
        for mode in self.modes:
            print(mode["symmetries"])
        



        

        # Display the plot
        return fig
    
    @staticmethod
    def basic_geometry(radius_1=0.2,  
                       eps_atom_1=1, 
                       radius_2=None, 
                       eps_atom_2=None,
                       eps_bulk = 12, 
                       ):
        geometry = [
            mp.Block(
                size = mp.Vector3(mp.inf, mp.inf),
                material=mp.Medium(epsilon=eps_bulk)),
            ]
        if radius_2 is None:
            geometry.append(mp.Cylinder(radius_1, 
                                        material=mp.Medium(epsilon=eps_atom_1),
                                        center = mp.Vector3(0,0)))
        else:
            geometry.append(mp.Cylinder(radius_1, 
                                        material=mp.Medium(epsilon=eps_atom_1),
                                        center = mp.Vector3(-0.5,-0.5)))
            geometry.append(mp.Cylinder(radius_2, 
                                        material=mp.Medium(epsilon=eps_atom_2),
                                        center = mp.Vector3(.5,.5)))
        return geometry
    
    @staticmethod
    def basic_lattice(lattice_type='square'):
        if lattice_type == 'square':
            return Crystal2D.square_lattice()
        elif lattice_type == 'triangular':
            return Crystal2D.triangular_lattice()
        else:
            raise ValueError("Invalid lattice type. Choose 'square' or 'triangular'.")
        
    @staticmethod
    def square_lattice():
        """
        Define the square lattice for the photonic crystal.
        
        Returns:
        - lattice: The lattice object representing the square lattice.
        - k_points: A list of k-points for the simulation.
        """
        lattice = mp.Lattice(size=mp.Vector3(1, 1),
                          basis1=mp.Vector3(1, 0),
                          basis2=mp.Vector3(0, 1))
        k_points = [
            mp.Vector3(),               # Gamma
            mp.Vector3(y=0.5),          # M
            mp.Vector3(0.5, 0.5),       # X
            mp.Vector3(),               # Gamma
        ]
        return lattice, k_points

    @staticmethod
    def triangular_lattice():
        """
        Define the triangular lattice for the photonic crystal.
        
        Returns:
        - lattice: The lattice object representing the triangular lattice.
        - k_points: A list of k-points for the simulation.
        """
        lattice = mp.Lattice(size=mp.Vector3(1, 1),
                          basis1=mp.Vector3(1, 0),
                          basis2=mp.Vector3(0.5, math.sqrt(3)/2))
        k_points = [
            mp.Vector3(),               # Gamma
            mp.Vector3(y=0.5),          # K
            mp.Vector3(-1./3, 1./3),    # M
            mp.Vector3(),               # Gamma
        ]
        return lattice, k_points
    
       
    
    


class CrystalSlab(PhotonicCrystal):
    def __init__(self,
                lattice_type = None,
                num_bands: int = 4,
                resolution = mp.Vector3(32,32,16),
                interp: int =2,
                periods: int =3, 
                pickle_id = None,
                geometry = None):
        super().__init__(lattice_type, num_bands, resolution, interp, periods, pickle_id)
        
        
        self.geometry_lattice, self.k_points = self.basic_lattice(lattice_type)
        self.geometry = geometry if geometry is not None else self.basic_geometry()
        self.k_points_interpolated = mp.interpolate(interp, self.k_points)

    def plot_epsilon_interactive(self, 
                                 fig=None, 
                                 title='Epsilon', 
                                 opacity=0.3, 
                                 colorscale='PuBuGn', 
                                 override_resolution_with: None|int= None, 
                                 periods = 1):
        """
        Plot the epsilon values obtained from the simulation interactively using Plotly.
        """

        if self.epsilon is None:
            
            if override_resolution_with is None:
                resolution = self.resolution
            else:
                resolution = override_resolution_with
            md = mpb.MPBData(rectify=True, periods=periods, resolution=resolution)
            
            
            converted_eps = md.convert(self.ms.get_epsilon())
        else:
            converted_eps = self.epsilon
        self.epsilon = converted_eps
        if fig is None:
            fig = go.Figure()

        z_points = converted_eps.shape[2]//periods
        z_mid = converted_eps.shape[2]//2
        epsilon = converted_eps[..., z_mid-z_points//2:z_mid+z_points//2-1]  

        # Create indices for x, y, z axes (meshgrid)
        x, y, z = np.meshgrid(np.arange(epsilon.shape[0]),
                            np.arange(epsilon.shape[1]),
                            np.arange(epsilon.shape[2]))

        # Flatten the arrays for Plotly
        x_flat = x.flatten()
        y_flat = y.flatten()
        z_flat = z.flatten()
        epsilon_flat = epsilon.flatten()

        # Get the minimum and maximum values from the epsilon array (ensure they are floats)
        isomin_value = float(np.min(epsilon_flat))
        isomax_value = float(np.max(epsilon_flat))

        # Create the 3D volume plot using Plotly
        fig = go.Figure(data=go.Volume(
            x=x_flat, y=y_flat, z=z_flat,
            value=epsilon_flat,  # Use the dielectric function values
            isomin=isomin_value,
            isomax=isomax_value,
            opacity=opacity,  # Adjust opacity to visualize internal structure
            surface_count=3,  # Number of surfaces to display
            colorscale=colorscale,  # Color scale for the dielectric function
            colorbar=dict(title='Dielectric Constant')
        ))

        # Add layout details
        fig.update_layout(
            title='3D Volume Plot of Dielectric Function',
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
            )
        )

        return fig

    @staticmethod
    def basic_lattice(lattice_type='square', height_supercell=4):
        if lattice_type == 'square':
            return CrystalSlab.square_lattice()
        elif lattice_type == 'triangular':
            return CrystalSlab.triangular_lattice()
        else:
            raise ValueError("Invalid lattice type. Choose 'square' or 'triangular'.")
        
    @staticmethod
    def square_lattice(height_supercell=4):
        """
        Define the square lattice for the photonic crystal.
        
        Returns:
        - lattice: The lattice object representing the square lattice.
        - k_points: A list of k-points for the simulation.
        """
        lattice = mp.Lattice(size=mp.Vector3(1, 1, height_supercell),
                          basis1=mp.Vector3(1, 0),
                          basis2=mp.Vector3(0, 1))
        k_points = [
            mp.Vector3(),               # Gamma
            mp.Vector3(y=0.5),          # M
            mp.Vector3(0.5, 0.5),       # X
            mp.Vector3(),               # Gamma
        ]
        return lattice, k_points

    @staticmethod
    def triangular_lattice(height_supercell=4):
        """
        Define the triangular lattice for the photonic crystal.
        
        Returns:
        - lattice: The lattice object representing the triangular lattice.
        - k_points: A list of k-points for the simulation.
        """
        lattice = mp.Lattice(size=mp.Vector3(1, 1, height_supercell),
                          basis1=mp.Vector3(1, 0),
                          basis2=mp.Vector3(0.5, math.sqrt(3)/2))
        k_points = [
            mp.Vector3(),               # Gamma
            mp.Vector3(y=0.5),          # K
            mp.Vector3(-1./3, 1./3),    # M
            mp.Vector3(),               # Gamma
        ]
        return lattice, k_points
    
    @staticmethod
    def basic_geometry(radius_1=0.2,  
                       eps_atom_1=1, 
                       radius_2=None, 
                       eps_atom_2=None,
                       eps_bulk = 12,
                       height_supercell=4, 
                       height_slab=0.5,
                       eps_background=1,
                       eps_substrate=None,
                       ):
        
        geometry = []
        
        #background
        geometry.append(
            mp.Block(
                size = mp.Vector3(mp.inf, mp.inf, height_supercell),
                material=mp.Medium(epsilon=eps_background)),
            
        )
        
        #substrate
        if eps_substrate is not None:
            geometry.append(mp.Block(
                size = mp.Vector3(mp.inf, mp.inf, height_supercell*0.5),
                center = mp.Vector3(0, 0, -height_supercell*0.25),
                material=mp.Medium(epsilon=eps_substrate)))

        
        #slab
        geometry.append(
            mp.Block(
                size = mp.Vector3(mp.inf, mp.inf, height_slab),
                material=mp.Medium(epsilon=eps_bulk)),
        )

        #atoms    
        if radius_2 is None:
            geometry.append(mp.Cylinder(radius_1, 
                                        material=mp.Medium(epsilon=eps_atom_1),
                                        height=height_slab))
                
            
        else:
            geometry.append(mp.Cylinder(radius_1, 
                                        material=mp.Medium(epsilon=eps_atom_1),
                                        height=height_slab))
            geometry.append(mp.Cylinder(radius_2, 
                                        material=mp.Medium(epsilon=eps_atom_2),
                                        height=height_slab))
        
        return geometry
    
    
    def plot_field_interactive(self, 
                               runner="run_tm", 
                               k_point=mp.Vector3(0, 0), 
                               periods=5, 
                               fig=None,
                               title="Field Visualization", 
                               colorscale='RdBu'):
            raw_fields = []
            freqs = []

            ms = mpb.ModeSolver(geometry=self.geometry,
                                geometry_lattice=self.geometry_lattice,
                                k_points=[k_point],
                                resolution=self.resolution,
                                num_bands=self.num_bands)
            
            def get_zodd_fields(ms, band):
                raw_fields.append(ms.get_hfield(band, bloch_phase=True))
            def get_zeven_fields(ms, band):
                raw_fields.append(ms.get_efield(band, bloch_phase=True))
            def get_freqs(ms, band):
                freqs.append(ms.freqs[band-1])

            
            with suppress_output():
                if runner == "run_te" or runner == "run_zeven":
                    ms.run_te(mpb.output_at_kpoint(k_point, mpb.fix_hfield_phase, get_zodd_fields, get_freqs))
                    field_type = "H-field"
                    # print(f"frequencies: {freqs}")
                    
                elif runner == "run_tm" or runner == "run_zodd":
                    ms.run_tm(mpb.output_at_kpoint(k_point, mpb.fix_efield_phase, get_zeven_fields, get_freqs))
                    field_type = "E-field"
                    
                    # print(f"frequencies: {freqs}")
                    
                else:
                    raise ValueError("Invalid runner. Please enter 'run_te', 'run_zeven', or 'run_tm' or 'run_zodd'.")
            
                md = mpb.MPBData(rectify=True, periods=periods)
                eps = md.convert(ms.get_epsilon())
                
                z_points = eps.shape[2]//periods
                z_mid = eps.shape[2]//2
                
                #now take only epsilon in the center of the slab
                eps = eps[..., z_mid]
            

                fields = []        
                for field in raw_fields:
                    
                    field = field[..., z_points // 2, 2]  # Get just the z component of the fields in the center
                    fields.append(md.convert(field))
                
                
                
                # print(f"Epsilon shape: {eps.shape}")
                # print(f"Field shape: {fields[-1].shape}")
                num_plots = len(fields)
                if num_plots == 0:
                    print("No field data to plot.")
                    return

            if fig is None:
                fig = go.Figure()

            # Automatically generate the subtitle with the k-vector and field type
            subtitle = f"{field_type}, z-component<br>k = ({k_point.x:.4f}, {k_point.y:.4f})"

            # Initialize an empty list for dropdown menu options
            dropdown_buttons = []

            # Calculate the midpoint between min and max of the permittivity (eps)
            min_eps, max_eps = np.min(eps), np.max(eps)
            midpoint = (min_eps + max_eps) / 2  # The level to be plotted

            for i, (field, freq) in enumerate(zip(fields, freqs)):
                visible_status = [False] * (2 * num_plots)
                visible_status[2 * i] = True  # Make the current contour (eps) visible
                visible_status[2 * i + 1] = True  # Make the current heatmap (field) visible

                # Add the contour plot for permittivity (eps) at the midpoint
                fig.add_trace(go.Contour(z=eps.T,
                                        contours=dict(
                                            start=midpoint,  # Start and end at the midpoint to ensure a single level
                                            end=midpoint,
                                            size=0.1,  # A small size to keep it as a single contour
                                            coloring='none'  # No filling
                                        ),
                                        line=dict(color='black', width=2),
                                        showscale=False,
                                        opacity=0.7,
                                        visible=True if i == 0 else False))  # Initially visible only for the first plot
                
                # Add the heatmap for the real part of the electric field
                fig.add_trace(go.Heatmap(z=np.real(field).T, colorscale=colorscale, zsmooth='best', opacity=0.9,
                                        showscale=False, visible=True if i == 0 else False))

                # Create a button for each field dataset for the dropdown
                dropdown_buttons.append(dict(label=f'Mode {i + 1}',
                                            method='update',
                                            args=[{'visible': visible_status},  # Update visibility for both eps and field
                                                {'title':f"{title}<br>Mode {i + 1}, freq={freq:0.3f}, z=0: {subtitle}"}
                                            ]))

            # Add the dropdown menu to the layout
            fig.update_layout(
                updatemenus=[dict(active=0,  # The first dataset is active by default
                                buttons=dropdown_buttons,
                                x=1.15, y=1.15,  # Positioning the dropdown to the top right
                                xanchor='left', yanchor='top')],
                title=f"{title}<br>Mode {1}, freq={freqs[0]:0.3f}, z=0: {subtitle}",  # Main title + subtitle
                xaxis_showgrid=False, yaxis_showgrid=False,
                xaxis_zeroline=False, yaxis_zeroline=False,
                xaxis_visible=False, yaxis_visible=False,
                hovermode="closest",
                width=800, height=800
            )

            # Display the plot
            return fig



#%%
def main():
    pass

def test_slab():
    #%%
    
    from photonic_crystal import CrystalSlab
    
    

    # Define basic parameters
    geometry = Crystal2D.basic_geometry(radius_1=0.35, eps_atom_1=1, eps_bulk=12)
    num_bands = 6
    resolution = 32
    interp = 4
    periods = 3
    lattice_type = 'square'
    pickle_id = 'test_crystal_2d'

    # Create an instance of the Crystal2D class
    crystal_2d = Crystal2D(lattice_type=lattice_type, 
                           num_bands=num_bands, 
                           resolution=resolution, 
                           interp=interp, 
                           periods=periods, 
                           pickle_id=pickle_id,
                           geometry=geometry)

    # Set the solver
    crystal_2d.set_solver()
    crystal_2d.run_simulation(runner="run_tm")
    print("Dummy simulation run")

    # Extract data
    crystal_2d.extract_data(periods=1)
    print("Data extracted")

    # Plot epsilon interactively
    print("Start plotting epsilon")
    fig_eps = crystal_2d.plot_epsilon_interactive()
    print("Ready to show epsilon plot")
    fig_eps.show()

    # Plot bands interactively
    print("Start plotting bands")
    fig_bands = crystal_2d.plot_bands_interactive(polarization="tm", title='Bands', color='blue')
    print("Ready to show bands plot")
    fig_bands.show()

 
    

 
    
    #%%

if __name__ == "__main__":
    main()
# %%
