import math
import meep as mp
from meep import mpb
import matplotlib.pyplot as plt  # Ensure you have matplotlib imported for plotting
import sys 
import contextlib
import plotly.graph_objects as go
import plotly.subplots
import numpy as np
import pickle 


class PhotonicCrystal:
    
    def __init__(self, 
                geometry=None,  
                geometry_lattice=None, 
                num_bands=8,
                resolution=32,
                k_points=None,
                interp=10,
                periods=5, 
                lattice_type=None,
                pickle_id = None):

        """
        Initialize a PhotonicCrystal object.

        Parameters:
        geometry (optional): The geometry of the photonic crystal. Defaults to None.
        geometry_lattice (optional): The lattice geometry of the photonic crystal. Defaults to None.
        num_bands (int, optional): The number of bands to compute. Defaults to 8.
        resolution (int, optional): The resolution of the simulation. Defaults to 32.
        k_points (optional): The k-points for the simulation. Defaults to None.
        interp (int, optional): The interpolation factor for k-points. Defaults to 4.
        lattice_type (optional): The type of lattice ('square' or 'triangular'). Defaults to None.
        periods (int, optional): The number of periods to extract. Defaults to 5.
        pickle_id (optional): The pickle ID for the simulation. Defaults to None.

        Raises:
        ValueError: If geometry_lattice is None and k_points is not None.
        """

        if geometry is None:
            geometry = PhotonicCrystal.basic_geometry()


        if geometry_lattice is None and k_points is not None:
            raise ValueError("Both geometry_lattice and k_points must be None or both not None.")
         
        if geometry_lattice is not None and k_points is None:
            raise ValueError("Both geometry_lattice and k_points must be None or both not None.")

        
        if lattice_type is None or lattice_type == 'square':
                geometry_lattice, k_points = PhotonicCrystal.square_lattice()
                print("square lattice")
                self.lattice_type = 'square'
        elif lattice_type == 'triangular':
                geometry_lattice, k_points = PhotonicCrystal.triangular_lattice()
                print("triangular lattice")
                self.lattice_type = 'triangular'
        

        self.num_bands = num_bands
        self.resolution = resolution
        self.periods = periods
        self.geometry_lattice = geometry_lattice
        self.geometry = geometry
        self.k_points = k_points
        self.k_points_interpolated = mp.interpolate(interp, self.k_points)


        self.freqs_te = None
        self.freqs_tm = None
        self.freqs = None
        self.gaps_te = None
        self.gaps_tm = None
        self.gaps = None

        self.md = None
        self.converted_eps = None

        # Allow the user to pickle the object if an ID is provided
        if pickle_id:
            self.pickle_photonic_crystal(pickle_id)

    def pickle_photonic_crystal(self, pickle_id):
        # Ensure the 'pickle_data' directory exists
        if not os.path.exists('pickle_data'):
            os.makedirs('pickle_data')

        # Save the object to a pickle file
        with open(f'pickle_data/{pickle_id}.pkl', 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load_photonic_crystal(pickle_id):
        # Load the object from the pickle file
        with open(f'pickle_data/{pickle_id}.pkl', 'rb') as file:
            return pickle.load(file)
        
    
    def set_solver(self):
        """
        Set the solver for the simulation.

        Call this method before running asimulation.
        """
        self.ms = mpb.ModeSolver(
            geometry=self.geometry,
            geometry_lattice=self.geometry_lattice,
            k_points=self.k_points_interpolated,
            resolution=self.resolution,
            num_bands=self.num_bands
        )
    def run_simulation(self, type='both', runner=None):
        """
        Run the simulation to calculate the frequencies and gaps.
        
        Parameters:
        - type: The polarization type ('tm' or 'te' or 'both'). Default is 'both'.
        - runner: The name of the function to run the simulation. Default is None
        """

        with suppress_output():

            self.set_solver()


            if runner is None: 
                if type == 'tm':
                    self.ms.run_tm()
                    self.freqs_tm = self.ms.all_freqs
                    self.gaps_tm = self.ms.gap_list
                elif type == 'te':
                    self.ms.run_te()
                    self.freqs_te = self.ms.all_freqs
                    self.gaps_te = self.ms.gap_list
                elif type == 'both':
                    self.ms.run_tm()
                    self.freqs_tm = self.ms.all_freqs
                    self.gaps_tm = self.ms.gap_list

                    self.ms.run_te()
                    self.freqs_te = self.ms.all_freqs
                    self.gaps_te = self.ms.gap_list
                else:
                    print("Invalid type. Please enter 'tm', 'te' or 'both'.")
            else:
                getattr(self.ms, runner)()
                self.freqs = self.ms.all_freqs
                self.gaps = self.ms.gap_list
            
        
                    

    def extract_data(self, periods: int | None = 5):
        """
        Extract the data from the simulation.
        
        Parameters:
        - periods_: The number of periods to extract. Default is 3.
        """
        self.md = mpb.MPBData(rectify=True, periods=periods, resolution=self.resolution)

    def plot_epsilon(self, fig=None, ax=None, title='Epsilon'):
        """
        Plot the epsilon values obtained from the simulation.

        """

        if periods is None:
            periods = self.periods


        with suppress_output():
            if self.ms is None:
                print("Simulation not run yet. Please run the simulation first.")
                return
            eps = self.ms.get_epsilon()
            converted_eps = self.md.convert(eps)
            
            if ax is not None:
                im = ax.imshow(converted_eps, interpolation='spline36', cmap='viridis')
                ax.axis('off')
                ax.set_title(title)
                cbar = fig.colorbar(im, ax=ax)
                cbar.set_label('$\\epsilon $')

            else:
                plt.imshow(converted_eps, interpolation='spline36', cmap='viridis')
                cbar = plt.colorbar()
                cbar.set_label('$\\epsilon $')
                plt.axis('off')
                plt.title(title)
            
            self.converted_eps = converted_eps
            return converted_eps

    def plot_epsilon_interactive(self, fig=None, title='Epsilon'):
        """
        Plot the epsilon values obtained from the simulation interactively using Plotly.
        """
        with suppress_output():
            if self.ms is None:
                print("Simulation not run yet. Please run the simulation first.")
                return
            eps = self.ms.get_epsilon()
            converted_eps = self.md.convert(eps)

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
             
            self.converted_eps = converted_eps
            return converted_eps
            


        

    def plot_bands(self, 
                   fig=None, 
                   ax=None, 
                   polarization="te", 
                   color= "red",  
                   title='Bands'):
        """
        Plot the band structure of the photonic crystal.
        """
        
        freqs = getattr(self, f'freqs_{polarization}')
        gaps = getattr(self, f'gaps_{polarization}')
        if freqs is None:
            print("Simulation not run yet. Please run the simulation first.")
            return
        xs = range(len(freqs))
        if ax is not None:
            ax.plot(freqs, color=color)

            for gap in gaps:
                if gap[0]>1:
                    ax.fill_between(xs, gap[1], gap[2], color= color, alpha=0.2)
            
            for x,freq in zip(xs,freqs):
                ax.scatter([x]*len(freq), freq, color=color, s=10)
        
        
        

        points_in_between = (len(freqs) - 4) / 3
        tick_locs = [i*points_in_between+i for i in range(4)]
        tick_labs = ['Γ', 'X', 'M', 'Γ']
        ax.set_xticks(tick_locs)
        ax.set_xticklabels(tick_labs, size=16)
        ax.set_ylabel('frequency (c/a)', size=16)
        ax.grid(True)
        ax.set_xlim(0, len(freqs)-1)


    def plot_bands_interactive(self, polarization="te", title='Bands', fig=None, color='blue'):
        """
        Plot the band structure of the photonic crystal interactively using Plotly, 
        with k-points displayed on hover and click events to trigger external scripts.
        """
        freqs = getattr(self, f'freqs_{polarization}')
        gaps = getattr(self, f'gaps_{polarization}')
        if freqs is None:
            print("Simulation not run yet. Please run the simulation first.")
            return

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
            name=f'{polarization.upper()} polarization'
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
        """
        Get the high symmetry points for the simulation.
        """
        k_high_sym = {}
        if self.lattice_type == 'square':
            k_high_sym['Γ'] = self.k_points[0]  # Gamma
            k_high_sym['X'] = self.k_points[2]
            k_high_sym['M'] = self.k_points[1]
        elif self.lattice_type == 'triangular':
            k_high_sym['Γ'] = self.k_points[0]  # Gamma
            k_high_sym['K'] = self.k_points[1]
            k_high_sym['M'] = self.k_points[2]
        else:
            print("Lattice type was not specified")
        
        return k_high_sym

        

        
    @staticmethod
    def basic_geometry(radius_1=0.2, eps_block = 12,  eps_atom_1=1, radius_2=None, eps_atom_2=None,  height=1):
        """
        Define the basic geometry of the photonic crystal.
        
        Returns:
        - geometry: A list of geometric objects representing the crystal structure.
        """
        if radius_2 is None:
            radius_2 = radius_1 
        if eps_atom_2 is None:
            eps_atom_2 = eps_atom_1
        if height==0.0:
            height=mp.inf
        
        geometry = [
            mp.Block(mp.Vector3(mp.inf, mp.inf, height), material=mp.Medium(epsilon=eps_block), center=mp.Vector3(0, 0)),
            mp.Cylinder(radius=radius_1, material=mp.Medium(epsilon=eps_atom_1), center=mp.Vector3(0, 0)),
        ]
        return geometry
    
    @staticmethod
    def slab_geometry(slab_h=1, 
                      supercell_h=4,
                      eps_sub=12,
                      eps_atom_1=1,
                      eps_atom_2=1,
                      eps_background = 1,
                      radius_1=0.2,
                      radius_2=0.2
    ):
        
        """
        Define the slab geometry for the photonic crystal.
        
        Returns:
        - geometry: A list of geometric objects representing the slab structure.
        """
        geometry = [
            #background
            mp.Block(material = mp.Medium(epsilon=eps_background),
                     center = mp.Vector3(0, 0,0.5*supercell_h),
                     size = mp.Vector3(mp.inf, mp.inf, supercell_h)),
            #slab
            mp.Block(material = mp.Medium(epsilon=eps_sub),
                    center = mp.Vector3(0, 0, 0.5*supercell_h),
                    size = mp.Vector3(mp.inf, mp.inf, slab_h)),
            #atoms
            mp.Cylinder(radius=radius_1, material=mp.Medium(epsilon=eps_atom_1), height=supercell_h, center=mp.Vector3(0, 0)),
            mp.Cylinder(radius=radius_2, material=mp.Medium(epsilon=eps_atom_2), height=supercell_h, center=mp.Vector3(0, 0)),
        ]
        return geometry
                    
    @staticmethod
    def square_lattice(supercell_h=4):
        """
        Define the square lattice for the photonic crystal.
        
        Returns:
        - lattice: The lattice object representing the square lattice.
        - k_points: A list of k-points for the simulation.
        """
        lattice = mp.Lattice(size=mp.Vector3(1, 1, supercell_h),
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
    def triangular_lattice(supercell_h = 4):
        """
        Define the triangular lattice for the photonic crystal.
        
        Returns:
        - lattice: The lattice object representing the triangular lattice.
        - k_points: A list of k-points for the simulation.
        """
        lattice = mp.Lattice(size=mp.Vector3(1, 1, supercell_h),
                          basis1=mp.Vector3(1, 0),
                          basis2=mp.Vector3(0.5, math.sqrt(3)/2))
        k_points = [
            mp.Vector3(),               # Gamma
            mp.Vector3(y=0.5),          # K
            mp.Vector3(-1./3, 1./3),    # M
            mp.Vector3(),               # Gamma
        ]
        return lattice, k_points
    
    def plot_field(self, runner = "run_tm", band=1, k_point=mp.Vector3(1 / -3, 1 / 3), periods=3, resolution=32):
        """
        Plot the electric field for a given band and k-point.
        
        Parameters:
        - runner: The name of the function to run the simulation. Default is "run_tm".
        - band: The band index to plot. Default is 1.
        - k_point: The k-point to plot the field at. Default is mp.Vector3(1 / -3, 1 / 3).
        - periods: The number of periods to extract. Default is 3.
        - resolution: The resolution of the field plot. Default is 32.



        """
        efields = []

        def get_efields(ms, band):
            efields.append(ms.get_efield(band, bloch_phase=True))

        with suppress_output():
            if runner is None:
                self.ms.run_tm(mpb.output_at_kpoint(k_point, mpb.fix_efield_phase, get_efields))
            else:
                getattr(self.ms, runner)(mpb.output_at_kpoint(k_point, mpb.fix_efield_phase, get_efields))

        print(f"ms:{self.ms}")
        md = self.md
        converted = []
        for f in efields:
            f = f[..., 0, 2]  # Get just the z component of the efields
            converted.append(md.convert(f))
        eps = md.convert(self.ms.get_epsilon())
        for i, f in enumerate(converted):
            plt.subplot(331 + i)
            plt.contour(eps.T, cmap='binary')
            plt.imshow(np.real(f).T, interpolation='spline36', cmap='RdBu', alpha=0.9)
            plt.axis('off')

        

    def plot_field_interactive(self, 
                           runner="run_tm", 
                           k_point=mp.Vector3(1 / -3, 1 / 3), 
                           frequency=None,
                           periods=5, 
                           resolution=32, 
                           fig=None,
                           title="Field Visualization",  # Default main title
                           colorscale='RdBu'):
        """
        Plot the electric field for a given band and k-point interactively using Plotly with a dropdown menu.
        
        Parameters:
        - runner: The name of the function to run the simulation. Default is "run_tm".
        - k_point: The k-point to plot the field at. Default is mp.Vector3(1 / -3, 1 / 3).
        - periods: The number of periods to extract. Default is 3.
        - resolution: The resolution of the field plot. Default is 32.
        - fig: The Plotly figure to plot on. Default is None.
        - title: The main title of the plot. Defaults to "Field Visualization".
        - colorscale: The colorscale to use for the plot. Default is 'RdBu'.
        
        """
        fields = []
        freqs = []
        self.set_solver()
        def get_hfields(ms, band):
            fields.append(ms.get_hfield(band, bloch_phase=True))
        def get_efields(ms, band):
            fields.append(ms.get_efield(band, bloch_phase=True))
        def get_freqs(ms, band):
            freqs.append(ms.freqs[band-1])

        self.ms.get_freqs
        with suppress_output():
            if runner == "run_te":
                self.ms.run_te(mpb.output_at_kpoint(k_point, mpb.fix_hfield_phase, get_hfields, get_freqs))
                field_type = "H-field"
                
            elif runner == "run_tm":
                self.ms.run_tm(mpb.output_at_kpoint(k_point, mpb.fix_efield_phase, get_efields, get_freqs))
                field_type = "E-field"
                
            else:
                raise ValueError("Invalid runner. Please enter 'run_te' or 'run_tm'.")
            
            md = self.md

            converted = []        
            for f in fields:
                f = f[..., 0, 2]  # Get just the z component of the fields
                converted.append(md.convert(f))

            eps = md.convert(self.ms.get_epsilon())
        
        num_plots = len(converted)
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

        for i, (f, freq) in enumerate(zip(converted, freqs)):
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
            fig.add_trace(go.Heatmap(z=np.real(f).T, colorscale=colorscale, zsmooth='best', opacity=0.9,
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
            width=500, height=500
        )

        # Display the plot
        return fig




import sys
import os

# Function that prints to stdout
def noisy_function():
    print("This is printed to the terminal!")

# Context manager to suppress stdout
class suppress_output:
    def __enter__(self):
        # Save original stdout and stderr file descriptors
        self._stdout_fd = sys.stdout.fileno()
        self._stderr_fd = sys.stderr.fileno()

        # Duplicate stdout and stderr to restore later
        self._saved_stdout_fd = os.dup(self._stdout_fd)
        self._saved_stderr_fd = os.dup(self._stderr_fd)

        # Redirect stdout and stderr to devnull
        devnull = os.open(os.devnull, os.O_RDWR)
        os.dup2(devnull, self._stdout_fd)
        os.dup2(devnull, self._stderr_fd)
        os.close(devnull)

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original stdout and stderr file descriptors
        os.dup2(self._saved_stdout_fd, self._stdout_fd)
        os.dup2(self._saved_stderr_fd, self._stderr_fd)

        # Close the duplicate file descriptors
        os.close(self._saved_stdout_fd)
        os.close(self._saved_stderr_fd)


                             
    
# Example usage
#%%
from photonic_crystal import PhotonicCrystal
import math
import meep as mp
from meep import mpb
from matplotlib import pyplot as plt
from IPython.display import display, HTML
from ipywidgets import Dropdown, Output, VBox
import os
import plotly.graph_objects as go



if __name__ == "__main__":

    pc0 = PhotonicCrystal(lattice_type='square', num_bands=8)

    pc0.run_simulation(type='both')
    pc0.extract_data(periods=5)
    
    pc1 = PhotonicCrystal(lattice_type='triangular')
    pc1.run_simulation(type='both')
    pc1.extract_data(periods=5)
    
    # Set width and height to match the iframe size in your HTML
    width = 500  # The width of the iframe container in pixels
    height = 500  # The height of the iframe container in pixels

    # Generate the figures with the same width and height as the iframe
    fig0 = go.Figure(layout=go.Layout(width=width, height=height))
    fig1 = go.Figure(layout=go.Layout(width=width, height=height))
    fig2 = go.Figure(layout=go.Layout(width=width, height=height))
    fig3 = go.Figure(layout=go.Layout(width=width, height=height))
    fig4 = go.Figure(layout=go.Layout(width=width, height=height))
    fig5 = go.Figure(layout=go.Layout(width=width, height=height))
    #%%
    # Plot epsilon interactively using Plotly
    converted_eps0 = pc0.plot_epsilon_interactive(title='Square Lattice', fig=fig0)
    print("Shape of converted_eps0:", converted_eps0.shape)
    fig0.show()
    converted_eps1 = pc1.plot_epsilon_interactive(title='Triangular Lattice', fig=fig1) 
    #%%
    # Plot bands interactively using Plotly
    pc0.plot_bands_interactive(polarization='te', title='Square Lattice TE and TM Bands', color='blue', fig=fig2)
    pc0.plot_bands_interactive(polarization='tm', title='Square Lattice TE and TM Bands', color='red', fig=fig2)
    pc1.plot_bands_interactive(polarization='te', title='Triangular Lattice TE and TM Bands', color='blue', fig=fig3)
    pc1.plot_bands_interactive(polarization='tm', title='Triangular Lattice TE and TM Bands', color='red', fig=fig3)

    
    # Get the k-point for the M point from the dictionary
    k_points_dict0 = pc0.get_high_symmetry_points()
    k_points_dict1 = pc1.get_high_symmetry_points()
    
    
   
    # Plot the field at the M point for the two crystals
    fig4 = pc0.plot_field_interactive(runner="run_tm", k_point=k_points_dict0['M'], title='Square Lattice Field at M Point', fig=fig4)
    fig5 = pc1.plot_field_interactive(runner="run_tm", k_point=k_points_dict1['M'], title='Triangular Lattice Field at M Point', fig=fig5)

    # Create the 'pics' directory if it doesn't exist
    if not os.path.exists('pics'):
        os.makedirs('pics')
    
    
    
    # Save the figures as HTML files
    fig0.write_html('pics/square_lattice_epsilon.html')
    fig1.write_html('pics/triangular_lattice_epsilon.html')
    fig2.write_html('pics/square_lattice_bands.html')
    fig3.write_html('pics/triangular_lattice_bands.html')
    fig4.write_html('pics/square_lattice_field.html')
    fig5.write_html('pics/triangular_lattice_field.html')
    

     
    

     
    # %%
