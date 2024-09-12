import math
import meep as mp
from meep import mpb
import matplotlib.pyplot as plt  # Ensure you have matplotlib imported for plotting
import sys 
import contextlib
import plotly.graph_objects as go


class PhotonicCrystal:
    
    def __init__(self, 
                geometry=None,  
                geometry_lattice=None, 
                num_bands=8,
                resolution=32,
                k_points=None,
                interp=10,
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
        elif lattice_type is None or lattice_type == 'square':
                geometry_lattice, k_points = PhotonicCrystal.square_lattice()
        elif lattice_type == 'triangular':
                geometry_lattice, k_points = PhotonicCrystal.triangular_lattice()

        self.num_bands = num_bands
        self.resolution = resolution
        self.geometry_lattice = geometry_lattice
        self.geometry = geometry
        self.k_points = k_points
        self.freqs_te = None
        self.freqs_tm = None
        self.freqs = None
        self.gaps_te = None
        self.gaps_tm = None
        self.gaps = None
        self.md = None

        self.k_points = mp.interpolate(interp, self.k_points)

        self.ms = mpb.ModeSolver(
            geometry=self.geometry,
            geometry_lattice=self.geometry_lattice,
            k_points=self.k_points,
            resolution=self.resolution,
            num_bands=self.num_bands
        )
    

    def run_simulation(self, type='both', runner=None, out_file="std_out_red.txt"):
        """
        Run the simulation to calculate the frequencies and gaps.
        
        Parameters:
        - type: The polarization type ('tm' or 'te' or 'both'). Default is 'both'.
        - runner: The name of the function to run the simulation. Default is None.
        - out_file: The file to redirect the standard output to. Default is None.
        """
        with open(out_file, 'w') as f:
            with contextlib.redirect_stdout(f):
        
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
                
            
                

    def extract_data(self, periods: int=3):
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

        return converted_eps

    def plot_epsilon_interactive(self, fig=None, title='Epsilon'):
        """
        Plot the epsilon values obtained from the simulation interactively using Plotly.
        """
        if self.ms is None:
            print("Simulation not run yet. Please run the simulation first.")
            return
        eps = self.ms.get_epsilon()
        converted_eps = self.md.convert(eps)

        if fig is None:
            fig = go.Figure()

        fig.add_trace(go.Heatmap(z=converted_eps, colorscale='Viridis'))
        fig.update_layout(title=title, coloraxis_colorbar=dict(title='$\\epsilon $'))

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




if __name__ == "__main__":

    pc0 = PhotonicCrystal(lattice_type='square', num_bands=8)

    pc0.run_simulation(type='both', out_file='output1.txt')
    pc0.extract_data(periods=5)
    
    pc1 = PhotonicCrystal(lattice_type='triangular')
    pc1.run_simulation(type='both', out_file='output2.txt')
    pc1.extract_data(periods=5)
    
    # Set width and height to match the iframe size in your HTML
    width = 500  # The width of the iframe container in pixels
    height = 500  # The height of the iframe container in pixels

    # Generate the figures with the same width and height as the iframe
    fig0 = go.Figure(layout=go.Layout(width=width, height=height))
    fig1 = go.Figure(layout=go.Layout(width=width, height=height))
    fig2 = go.Figure(layout=go.Layout(width=width, height=height))
    fig3 = go.Figure(layout=go.Layout(width=width, height=height))

    # Plot epsilon interactively using Plotly
    converted_eps0 = pc0.plot_epsilon_interactive(title='Square Lattice', fig=fig0)
    converted_eps1 = pc1.plot_epsilon_interactive(title='Triangular Lattice', fig=fig1)

    # Plot bands interactively using Plotly
    pc0.plot_bands_interactive(polarization='te', title='Square Lattice TE and TM Bands', color='blue', fig=fig2)
    pc0.plot_bands_interactive(polarization='tm', title='Square Lattice TE and TM Bands', color='red', fig=fig2)
    pc1.plot_bands_interactive(polarization='te', title='Triangular Lattice TE and TM Bands', color='blue', fig=fig3)
    pc1.plot_bands_interactive(polarization='tm', title='Triangular Lattice TE and TM Bands', color='red', fig=fig3)
    
    # Create the 'pics' directory if it doesn't exist
    if not os.path.exists('pics'):
        os.makedirs('pics')

    # Save the figures as HTML files
    fig0.write_html('pics/square_lattice_epsilon.html')
    fig1.write_html('pics/triangular_lattice_epsilon.html')
    fig2.write_html('pics/square_lattice_bands.html')
    fig3.write_html('pics/triangular_lattice_bands.html')
    
    

    

     
    # %%
