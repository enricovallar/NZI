import math
import meep as mp
from meep import mpb
import matplotlib.pyplot as plt  # Ensure you have matplotlib imported for plotting
import sys 
import contextlib
import plotly.graph_objects as go
import plotly.subplots
import numpy as np


class PhotonicCrystal:
    
    def __init__(self, 
                geometry=None,  
                geometry_lattice=None, 
                num_bands=8,
                resolution=32,
                k_points=None,
                interp=10,
                periods=5, 
                lattice_type=None):

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
        elif lattice_type == 'triangular':
                geometry_lattice, k_points = PhotonicCrystal.triangular_lattice()
                print("triangular lattice")

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
            
        
                    

    def extract_data(self, periods: int | None = None):
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
        Plot the band structure of the photonic crystal interactively using Plotly.
        """
        freqs = getattr(self, f'freqs_{polarization}')
        gaps = getattr(self, f'gaps_{polarization}')
        if freqs is None:
            print("Simulation not run yet. Please run the simulation first.")
            return

        xs = list(range(len(freqs)))
        if fig is None:
            fig = go.Figure()

        for band in zip(*freqs):
            fig.add_trace(go.Scatter(x=xs, y=band, mode='lines', line=dict(color=color)))

        for gap in gaps:
            if gap[0] > 1:
                fig.add_shape(type="rect",
                              x0=xs[0], x1=xs[-1],
                              y0=gap[1], y1=gap[2],
                              fillcolor=color, opacity=0.2, line_width=0)

        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(color=color),
            showlegend=True,
            name=polarization.upper()
        ))


        fig.update_layout(
            title=title,
            xaxis=dict(
                tickmode='array',
                tickvals=[i * (len(freqs) - 4) / 3 + i for i in range(4)],
                ticktext=['Γ', 'X', 'M', 'Γ']
            ),
            yaxis_title='frequency (c/a)',
            showlegend=False
        )

        
                

    def get_k_points_dictionary(self):
        """
        Get the dictionary of k-points for the simulation.
        """
        
        return {
            'Gamma': self.k_points[0],
            'M': self.k_points[1],
            'X': self.k_points[2],
        }

        
    @staticmethod
    def basic_geometry(radius=0.2, eps=12):
        """
        Define the basic geometry of the photonic crystal.
        
        Returns:
        - geometry: A list of geometric objects representing the crystal structure.
        """
        geometry = [
            mp.Cylinder(radius=radius, material=mp.Medium(epsilon=eps)),
        ]
        return geometry
    
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
            mp.Vector3(y=0.5),          # M
            mp.Vector3(-1./3, 1./3),    # K
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
                            periods=3, 
                            resolution=32, 
                            fig=None,
                            title='E-field Components Visualization',
                            colorscale='RdBu'):
        """
        Plot the electric field for a given band and k-point interactively using Plotly with a dropdown menu.
        
        Parameters:
        - runner: The name of the function to run the simulation. Default is "run_tm".
        - k_point: The k-point to plot the field at. Default is mp.Vector3(1 / -3, 1 / 3).
        - periods: The number of periods to extract. Default is 3.
        - resolution: The resolution of the field plot. Default is 32.
        - fig: The Plotly figure to plot on. Default is None.
        - title: The title of the plot. Default is 'E-field Components Visualization with Dropdown'.
        - colorscale: The colorscale to use for the plot. Default is 'RdBu'.
        """
        fields = []
        

        self.set_solver()
        def get_hfields(ms, band):
            fields.append(ms.get_hfield(band, bloch_phase=True))
        def get_efields(ms, band):
            fields.append(ms.get_efield(band, bloch_phase=True))

        with suppress_output():
            if runner=="run_te":
                self.ms.run_te(mpb.output_at_kpoint(k_point, mpb.fix_hfield_phase, get_hfields))
                field = "H-field, z-component"
            elif runner=="run_tm":
                self.ms.run_tm(mpb.output_at_kpoint(k_point, mpb.fix_efield_phase, get_efields))
                field = "E-field, z-component"
            else:
                raise ValueError("Invalid runner. Please enter 'run_te' or 'run_tm'.")
                

            md = self.md

            converted = []        
            for f in fields:
                f = f[..., 0, 2]  # Get just the z component of the efields
                converted.append(md.convert(f))

            eps = md.convert(self.ms.get_epsilon())

        num_plots = len(converted)
        if num_plots == 0:
            print("No field data to plot.")
            return

        if fig is None:
            fig = go.Figure()

        # Initialize an empty list for dropdown menu options
        dropdown_buttons = []

        # Calculate the midpoint between min and max of the permittivity (eps)
        min_eps, max_eps = np.min(eps), np.max(eps)
        midpoint = (min_eps + max_eps) / 2  # The level to be plotted

        for i, f in enumerate(converted):
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
            
            # Add the heatmap for the real part of the electric field (if required)
            fig.add_trace(go.Heatmap(z=np.real(f).T, colorscale=colorscale, zsmooth='best', opacity=0.9,
                                showscale=False, visible=True if i == 0 else False))

            # Create a button for each field dataset for the dropdown
            dropdown_buttons.append(dict(label=f'Mode {i + 1}',
                                        method='update',
                                        args=[{'visible': visible_status},  # Update visibility for both eps and field
                                            {'title': f"{title}<br>{field} - Mode {i + 1}"}
                                        ]))

        # Add the dropdown menu to the layout
        fig.update_layout(
            updatemenus=[dict(active=0,  # The first dataset is active by default
                            buttons=dropdown_buttons,
                            x=1.15, y=1.15,  # Positioning the dropdown to the top right
                            xanchor='left', yanchor='top')],
            title=title,
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
    converted_eps1 = pc1.plot_epsilon_interactive(title='Triangular Lattice', fig=fig1) 
    #%%
    # Plot bands interactively using Plotly
    pc0.plot_bands_interactive(polarization='te', title='Square Lattice TE and TM Bands', color='blue', fig=fig2)
    pc0.plot_bands_interactive(polarization='tm', title='Square Lattice TE and TM Bands', color='red', fig=fig2)
    pc1.plot_bands_interactive(polarization='te', title='Triangular Lattice TE and TM Bands', color='blue', fig=fig3)
    pc1.plot_bands_interactive(polarization='tm', title='Triangular Lattice TE and TM Bands', color='red', fig=fig3)

    
    # Get the k-point for the M point from the dictionary
    k_points_dict0 = pc0.get_k_points_dictionary()
    k_points_dict1 = pc1.get_k_points_dictionary()
    
    
   
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
