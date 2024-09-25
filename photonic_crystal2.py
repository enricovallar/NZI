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

class PhotonicCrystal:
    def __init__(self,
                lattice_type = None,
                num_bands: int = 6,
                resolution: tuple[int, int] | int = 32,
                interp: int = 4,
                periods: int = 3, 
                pickle_id = None):
        
        self.lattice_type = lattice_type
        self.num_bands = num_bands
        self.resolution = resolution
        self.interp = interp
        self.periods = periods
        self.pickle_id = pickle_id

        #this values are set with basic lattice method
        self.geometry_lattice= None 
        self.k_points = None
        #slef.geometry_lattice, self.k_points = self.basic_lattice()

        #this values are set with basic geometry method
        self.basic_geometry = None
        #self.geometry = self.basic_geometry()


        self.ms = None
        self.md = None

        self.freqs = {}
        self.gaps = {}
        self.epsilon = None
        self.fields ={}

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

    def run_simulation(self, 
                       runner_1="run_zeven",
                       runner_2 = None):
        """
        Run the simulation to calculate the frequencies and gaps.
        
        Parameters:
        - type: The polarization type ('tm' or 'te' or 'both'). Default is 'both'.
        - runner: The name of the function to run the simulation. Default is None
        """
        if self.ms is None:
            raise ValueError("Solver is not set. Call set_solver() before running the simulation.")

        if runner_1.startswith("run_"):
            polarization_1 = runner_1[4:]
        else:
            polarization_1 = runner_1

        if runner_2 is not None:
            if runner_2.startswith("run_"):
                polarization_2 = runner_2[4:]
            else:
                polarization_2 = runner_2
        with suppress_output():
            
            getattr(self.ms, runner_1)()
            self.freqs[polarization_1] = self.ms.all_freqs
            self.gaps[polarization_1]  = self.ms.gap_list

            if runner_2 is not None:
                getattr(self.ms, runner_2)()
                self.freqs[polarization_2] = self.ms.all_freqs
                self.gaps[polarization_2]  = self.ms.gap_list

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
        with k-points displayed on hover and click events to trigger external scripts.
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

        # Iterate through each frequency band
        for band in zip(*freqs):
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
                customdata=[(kp.x, kp.y, kp.z) for kp in k_points_interpolated],  # Attach k-points as custom data (vector components)
                showlegend=False  # Hide from legend
            ))

        # Add bandgap shading
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
                    line_width=0
                )

        # Add legend entries for each polarization
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(color=color),
            name=f'{polarization.upper()}'
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
            showlegend=True
        )
        
        # Add a JavaScript callback to handle clicks
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

        self.ms.get_freqs
       
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
        print(fields)
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
            md = mpb.MPBData(rectify=True, periods=1, resolution=resolution)
            
            converted_eps = md.convert(self.ms.get_epsilon())
        else:
            converted_eps = self.epsilon
        self.epsilon = converted_eps
        if fig is None:
            fig = go.Figure()

        epsilon = np.array(converted_eps)  # If epsilon is an MPBArray, convert it to a NumPy array

        # Replicate the epsilon array along x and y axes using self.periods
        epsilon = np.tile(epsilon, (periods, periods, 1))

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
            surface_count=20,  # Number of surfaces to display
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



#%%
def main():
    pass

def test_slab():
    #%%
    
    from photonic_crystal2 import CrystalSlab
    
    

    # Define basic parameters
    geometry = CrystalSlab.basic_geometry(radius_1=0.35, eps_atom_1=1, eps_bulk=12, eps_background=1, eps_substrate=1.45**2)
    num_bands = 4
    resolution = mp.Vector3(32, 32, 16)
    interp = 2
    periods = 5
    lattice_type = 'square'
    pickle_id = 'test_crystal_slab'

    # Create an instance of the CrystalSlab class
    crystal_slab = CrystalSlab(lattice_type=lattice_type, 
                                num_bands=num_bands, 
                                resolution=resolution, 
                                interp=interp, 
                                periods=periods, 
                                pickle_id=pickle_id,
                                geometry=geometry)

    crystal_slab.run_dumb_simulation()
    print("dummy simulation runned")
    # Extract data

    crystal_slab.extract_data(periods=1)
    print("data extracted")

    # Plot epsilon interactively
    print("start plotting")
    fig_eps = crystal_slab.plot_epsilon_interactive(colorscale = "matter", 
                                                              opacity=0.2,
                                                              override_resolution_with=16, 
                                                              periods = 2)
    print("ready to show")
    
    fig_eps.show()

    

 
    
    #%%

if __name__ == "__main__":
    main()
# %%
