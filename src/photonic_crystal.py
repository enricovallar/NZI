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
from functools import partial



class PhotonicCrystal:
    """
    A class to represent a photonic crystal and perform simulations using MPB (MIT Photonic Bands).
    
    Attributes:
        lattice_type (str): The type of lattice (e.g., 'square', 'triangular').
        num_bands (int): The number of bands to calculate.
        resolution (tuple[int, int] | int): The resolution of the simulation.
        interp (int): The interpolation factor for k-points.
        periods (int): The number of periods for the simulation.
        k_points (list): The list of k-points for the simulation.
        use_XY (bool): Whether to use X and Y directions for the x-axis in plots.
        geometry_lattice (mp.Lattice): The lattice geometry.
        k_points_interpolated (list): The interpolated k-points.
        basic_geometry (list): The basic geometry of the photonic crystal.
        ms (mpb.ModeSolver): The mode solver for the simulation.
        md (mpb.MPBData): The MPB data object.
        freqs (dict): The calculated frequencies for different polarizations.
        gaps (dict): The calculated band gaps for different polarizations.
        epsilon (np.ndarray): The dielectric constant distribution.
        modes (list): The list of calculated modes.
        has_been_run (bool): Whether the simulation has been run.
    
    Methods:
        __getstate__(): Get the state for pickling.
        __setstate__(state): Set the state after unpickling.
        pickle_photonic_crystal(pickle_id): Pickle the photonic crystal object.
        load_photonic_crystal(pickle_id): Load a pickled photonic crystal object.
        set_solver(k_point): Set the mode solver for the simulation.
        run_simulation(runner, polarization): Run the simulation to calculate the frequencies and gaps.
        run_simulation_with_output(runner, polarization): Run the simulation and get mode data.
        run_dumb_simulation(): Run a dumb simulation to quickly extract some values.
        convert_mode_fields(mode, periods): Convert the mode fields to arrays for visualization.
        extract_data(periods): Extract the data from the simulation.
        plot_epsilon(fig, title): Plot the epsilon of the photonic crystal interactively using Plotly.
        plot_bands(polarization, title, fig, color): Plot the bands of the photonic crystal using Plotly.
        get_XY_k_points_near_gamma(distance): Get the relevant k-points near the gamma point for the X and Y directions.
        get_high_symmetry_points(): Get the high symmetry points for the photonic crystal lattice.
        plot_field(target_polarization, target_k_point, target_frequency, frequency_tolerance, k_point_max_distance, periods, component, quantity, colorscale): Plot the field visualization.
        plot_field_components(target_polarization, target_k_point, target_frequency, frequency_tolerance, k_point_max_distance, periods, quantity, colorscale): Plot the field components (Ex, Ey, Ez) and (Hx, Hy, Hz) for specific modes with consistent color scales.
        look_for_mode(polarization, k_point, freq, freq_tolerance, k_point_max_distance): Look for modes within the specified criteria.
        find_modes_symmetries(): Find the symmetries of the modes.
        plot_modes_vectorial_fields(modes, sizemode, names): Plot the vectorial fields of the modes.
        plot_mode_fields_normal_to_k(mode, k): Plot the fields perpendicular to the wavevector k for the mode.
        plot_vectorial_fields(fields, colorscales, names): Plot the vectorial fields of the modes.
        _field_to_cones(field, colorscale, sizemode, sizeref, clim): Convert a field to cones for visualization.
        _fields_to_cones(fields, colorscale, sizemode, sizeref, clim, colorscales): Convert a list of fields to cones for visualization.
        _calculate_field_norm_to_k(fields, k): Calculate the components of the field perpendicular to the wavevector k.
        _get_direction(k_vector): Determine the primary direction of the wavevector k.
        _calculate_effective_parameter(mode): Calculate the effective parameters of the mode.
        basic_geometry(): Define the basic geometry of the photonic crystal.
        basic_lattice(): Define the basic lattice of the photonic crystal.
    """
    def __init__(self,
                lattice_type = None,
                num_bands: int = 6,
                resolution: tuple[int, int] | int = 32,
                interp: int = 4,
                periods: int = 3, 
                pickle_id = None, 
                k_points = None,
                use_XY  = True
                ):
        """
        Initializes the PhotonicCrystal class with the given parameters.
                Args:
                    lattice_type (str, optional): Type of the lattice. Defaults to None.
                    num_bands (int, optional): Number of bands. Defaults to 6.
                    resolution (tuple[int, int] | int, optional): Resolution of the simulation. Defaults to 32.
                    interp (int, optional): Interpolation factor for k-points. Defaults to 4.
                    periods (int, optional): Number of periods. Defaults to 3.
                    pickle_id (str, optional): Identifier for pickling. Defaults to None.
                    k_points (list, optional): List of k-points. Defaults to None.
                    use_XY (bool, optional): Flag to use XY plane. Defaults to True.
                Attributes:
                    lattice_type (str): Type of the lattice.
                    num_bands (int): Number of bands.
                    resolution (tuple[int, int] | int): Resolution of the simulation.
                    interp (int): Interpolation factor for k-points.
                    periods (int): Number of periods.
                    pickle_id (str): Identifier for pickling.
                    has_been_run (bool): Flag indicating if the simulation has been run.
                    geometry_lattice (None): Geometry lattice, set with basic lattice method.
                    k_points (list): List of k-points.
                    k_points_interpolated (list): Interpolated k-points.
                    basic_geometry (None): Basic geometry, set with basic geometry method.
                    ms (None): Placeholder for ms attribute.
                    md (None): Placeholder for md attribute.
                    freqs (dict): Dictionary to store frequencies.
                    gaps (dict): Dictionary to store gaps.
                    epsilon (None): Placeholder for epsilon attribute.
                    modes (list): List to store modes.
                    use_XY (bool): Flag to use XY plane.
        """
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
        self.use_XY = use_XY

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
        """Pickle the photonic crystal object.

        Args:
            pickle_id (str): The identifier for the pickle file.
        """
        with open(f"{pickle_id}.pkl", "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_photonic_crystal(pickle_id) -> 'PhotonicCrystal':
        """Load a pickled photonic crystal object.

        Args:
            pickle_id (str): The identifier for the pickle file.

        Returns:
            PhotonicCrystal: The loaded photonic crystal object.
        """
        with open(f"{pickle_id}.pkl", "rb") as f:
            return pickle.load(f)

    def set_solver(self, k_point = None):
        """
        Set the mode solver for the simulation. 
        For how MPB works, it is better to call this method each time you want to run a simulation.
        This method initializes the mode solver (ms) with the geometry, geometry lattice, 
        k-points, resolution, and number of bands. If a specific k-point is provided, 
        the solver is set up for that k-point. Otherwise, it uses the interpolated k-points.

        Args:
            k_point (mp.Vector3, optional): The k-point for the simulation. Default is None.
        
        """

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

        Args:
            runner (str): The name of the function to run the simulation. Default is 'run_zeven'. 
            polarization (str, optional): The polarization of the simulation. Default is None. If None, it uses the runner name.

        runner must correspond to an MPB runner. For example: \n
        -'run_zeven': Run the simulation for even parity modes in z-axis.\n
        -'run_zodd': Run the simulation for odd parity modes in z-axis.\n
        -'run_tm': Run the simulation for transverse magnetic modes.\n
        -'run_te': Run the simulation for transverse electric modes.\n
        -'run': Do not consider symmetry.
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
        
        # This is a custom mpb output function that stores the fields and frequencies
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

    def run_simulation_with_output(self, runner="run_zeven", polarization=None):
        """
        Run the simulation and get mode data. Mode data are not stored in the crystal object, 
        but are returned as a list of dictionaries.

        Args:
            runner (str): The name of the function to run the simulation. Default is 'run_zeven'. 
            polarization (str, optional): The polarization of the simulation. Default is None. If None, it uses the runner name.

        runner must correspond to an MPB runner. For example: \n
        -'run_zeven': Run the simulation for even parity modes in z-axis.\n
        -'run_zodd': Run the simulation for odd parity modes in z-axis.\n
        -'run_tm': Run the simulation for transverse magnetic modes.\n
        -'run_te': Run the simulation for transverse electric modes.\n
        -'run': Do not consider symmetry.
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
        modes=[]
        # This is a custom mpb output function that stores the fields and frequencies
        def get_mode_data(ms, band):
            mode = {
                "h_field": ms.get_hfield(band, bloch_phase=True),
                "e_field": ms.get_efield(band, bloch_phase=True),
                "freq": ms.freqs[band-1],
                "k_point": ms.current_k,
                "polarization": polarization
            }
            modes.append(mode)
        with suppress_output():
            getattr(self.ms, runner)(get_mode_data)
        return modes
        
       


    def run_dumb_simulation(self) -> mpb.ModeSolver:    
        """
        Run a dumb simulation. 
        This is used to quickly extract some values from the simulation later. 
        """

        #run the simulation in the gamma point, find one mode
        self.ms = mpb.ModeSolver(geometry=self.geometry,
                                  geometry_lattice=self.geometry_lattice,
                                  k_points=[mp.Vector3()],
                                  resolution=self.resolution,
                                  num_bands=1)
        
        self.ms.run()
        ms = self.ms
        return ms
    
    def convert_mode_fields(self, mode, periods=1)-> tuple[mpb.MPBArray, mpb.MPBArray]:
        """
        Convert the mode fields to arrays for visualization.
        Apparently this is necessary to visualize the fields if crystal is restored from pickle.

        Args:
        - mode: The mode dictionary.
        - periods: The number of periods to extract. Default is 1.
        
        Returns:
        - e_field_array: The electric field array for visualization.
        - h_field_array: The magnetic field array for visualization.
        """

        with suppress_output():
            self.run_dumb_simulation()
            md = mpb.MPBData(rectify=True, periods=periods, lattice=self.ms.get_lattice())
            e_field_array = mpb.MPBArray(mode["e_field"], lattice=self.ms.get_lattice(), kpoint=mode["k_point"])
            h_field_array = mpb.MPBArray(mode["h_field"], lattice=self.ms.get_lattice(), kpoint=mode["k_point"])
            
            e_field =  md.convert(e_field_array)
            h_field =  md.convert(h_field_array)            
            return e_field, h_field 
        


    def extract_data(self, periods: int | None = 5):
        """
        Extract the data from the simulation.

        Args:
            periods (int, optional): The number of periods to extract. Default is 5.

        Returns:
            mpb.MPBData: The MPB data object.
        """

        if self.ms is None:
            raise ValueError("Solver is not set. Call set_solver() before extracting data.")

        self.md = mpb.MPBData(rectify=True, periods=periods, resolution=self.resolution)
        return self.md
        
    
    def plot_epsilon(self, fig=None, title='Epsilon'):
        """
        Plot the epsilon of the photonic crystal interactively using Plotly.
        Not implemented in the base class. Must be implemented in the derived class.

        Args:
            fig (go.Figure, optional): The Plotly figure to add the epsilon plot to. Default is None.
            title (str, optional): The title of the plot. Default is 'Epsilon'.

        Returns:
            go.Figure: The Plotly figure object.(Not woroking yet, error is thrown)
        """
        raise NotImplementedError

    def plot_bands(self, polarization="te", title='Bands', fig=None, color='blue')-> go.Figure:
        """
        Plot the bands of the photonic crystal using Plotly.
        This method plots the bands for the specified polarization.
        In Dash and Jupyther Notebook, the plot is interactive and data are shown on hover.        

        Args:
            polarization (str, optional): The polarization of the bands. Default is 'te'.
            title (str, optional): The title of the plot. Default is 'Bands'.
            fig (go.Figure, optional): The Plotly figure to add the bands plot to. Default is None.
            color (str, optional): The color of the bands. Default is 'blue'.

        Returns:
            go.Figure: The Plotly figure object.
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

        # Customize the x-axis 
        if self.use_XY is True:  # Use X and Y directions for the x-axis
            relevant_k_points = self.get_XY_k_points_near_gamma()
            fig.update_layout(
                title=title,
                xaxis=dict(
                    tickmode='array',
                    tickvals=[i * (len(freqs) - 3) / 2 + i for i in range(3)],
                    ticktext=list(relevant_k_points.keys())  # Only three values, no repetition
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
        else:
            relevant_k_points = self.get_high_symmetry_points() # Use high symmetry points for the x-axis 
                                                                # Gamma, X, M for square lattice and
                                                                # Gamma, K, M for triangular lattice                                                   
            fig.update_layout(
                title=title,
                xaxis=dict(
                    tickmode='array',
                    tickvals=[i * (len(freqs) - 4) / 3 + i for i in range(4)],
                    ticktext=list(relevant_k_points.keys()) + [list(relevant_k_points.keys())[0]]  # Repeat the first element at the end
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

        return fig
    

    def get_XY_k_points_near_gamma(self, distance = 0.1) -> dict:
        """
        Get the relevant k-points near the gamma point for the X and Y directions.
        This is useful for plotting the bands with the X and Y directions on the x-axis.

        Args:
            distance (float): The distance from the gamma point. Default is 0.1.

        Returns:
            dict: A dictionary with the relevant k-points for the X and Y directions.
            with the k-point names as keys and the k-point vectors as values.
        """

        if distance >= 0.5:
            raise ValueError("Distance must be less than 0.5")
        relevant_k_points = {
            'X': mp.Vector3(0.5, 0),
            'Γ': mp.Vector3(0, 0, 0),
            'Y': mp.Vector3(0,0.5, 0)
        }
        return relevant_k_points

    def get_high_symmetry_points(self) -> dict:
        """
        Get the high symmetry points for the photonic crystal lattice.
        This is useful for plotting the bands with the high symmetry points on the x-axis.

        Returns:
            dict: A dictionary with the high symmetry points for the lattice.
            with the k-point names as keys and the k-point vectors as values.
        """

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


    def plot_field(
        self, 
        target_polarization,
        target_k_point,
        target_frequency,
        frequency_tolerance=0.01,
        k_point_max_distance=None,
        periods=5,
        component: int = 2,
        quantity: str = "real",
        colorscale: str = "RdBu",
    )->tuple[go.Figure, go.Figure]: 
        """
        Plot the field visualization.
        Not implemented in the base class. Must be implemented in the derived class.

        Args:
            target_polarization (str): The polarization of the mode.
            target_k_point (tuple): The k-point of the mode.
            target_frequency (float): The frequency of the mode.
            frequency_tolerance (float): The tolerance for frequency similarity. Default is 0.01.
            k_point_max_distance (float, optional): The maximum distance for k-point similarity. Default is None.
            periods (int): The number of periods to extract. Default is 5.
            component (int): The component of the field to plot. Default is 2.
            quantity (str): The quantity of the field to plot. Default is "real".
            colorscale (str): The colorscale for the plot. Default is "RdBu".

        Returns:
            tuple[go.Figure, go.Figure]: A tuple containing the Plotly figures for the electric and magnetic fields.           

        """
        raise NotImplementedError("plot_field method not implemented yet.")

    
    def plot_field_components(
            self,
            target_polarization,
            target_k_point,
            target_frequency,
            frequency_tolerance=0.01,
            k_point_max_distance=None,
            periods: int = 1,
            quantity: str = "real",
            colorscale: str = 'RdBu',
            )-> tuple[go.Figure, go.Figure]:
        
        """
        Plot the field visualization.
        Not implemented in the base class. Must be implemented in the derived class.

        Args:
            target_polarization (str): The polarization of the mode.
            target_k_point (tuple): The k-point of the mode.
            target_frequency (float): The frequency of the mode.
            frequency_tolerance (float): The tolerance for frequency similarity. Default is 0.01.
            k_point_max_distance (float, optional): The maximum distance for k-point similarity. Default is None.
            periods (int): The number of periods to extract. Default is 1.
            quantity (str): The quantity of the field to plot. Default is "real".
            colorscale (str): The colorscale for the plot. Default is "RdBu".

        Returns:
            tuple[go.Figure, go.Figure]: A tuple containing the Plotly figures for the electric and magnetic fields.           
        """

        raise NotImplementedError("plot_field_components method not implemented yet.")
        



    def look_for_mode(self, polarization, k_point, freq,  freq_tolerance=0.01, k_point_max_distance = None):
        """
        Look for modes within the specified criteria.

        Args:
            polarization (str): The polarization of the mode.
            k_point (tuple): The k-point of the mode.
            freq (float): The frequency of the mode.
            freq_tolerance (float): The tolerance for frequency similarity.
            k_point_max_distance (float, optional): The maximum distance for k-point similarity. Default is None.

        Returns:
            list: A list of mode dictionaries that match the criteria.
        """

        target_modes = []
        if k_point_max_distance is None:
            for mode in self.modes:
                if mode["polarization"] == polarization and mode["k_point"] == k_point and abs(mode["freq"] - freq) <= freq_tolerance:
                    target_modes.append(mode)
        else:
            for mode in self.modes:
                if mode["polarization"] == polarization and np.linalg.norm(np.array(mode["k_point"]) - np.array(k_point)) <= k_point_max_distance and abs(mode["freq"] - freq) <= freq_tolerance:
                    target_modes.append(mode)
        return target_modes
        
    

    def find_modes_symmetries(self):
        """
        Find the symmetries of the modes.
        Not implemented yet. TODO
        """
        raise NotImplementedError("find_modes_symmetries method not implemented yet.")
    


    def plot_modes_vectorial_fields(self, modes, sizemode="scaled", names=["Electric Field", "Magnetic Field"]):
        """
        Plot the vectorial fields of the modes.

        Args:
            modes (list): The list of modes to plot.
            sizemode (str): The sizemode for the cones. Default is 'scaled'.
            names (list): The names of the fields. Default is ['Electric Field', 'Magnetic Field'].

        Returns:
            tuple: A tuple containing the Plotly figures for the electric and magnetic fields.
        """
        
        colorscales = ["blues", "reds", "greens", "purples", "oranges", "ylorbr"]
        
        h_fields = [mode["h_field"] for mode in modes]
        e_fields = [mode["e_field"] for mode in modes]

        max_norm_h = PhotonicCrystal._calculate_fields_max_norms(h_fields)
        max_norm_e = PhotonicCrystal._calculate_fields_max_norms(e_fields)
        
        e_sizeref = max_norm_e
        h_sizeref = max_norm_h

        e_clim = (0, max_norm_e)
        h_clim= (0, max_norm_h)


        e_fig = go.Figure()
        h_fig = go.Figure()
        
        e_cones = PhotonicCrystal._fields_to_cones(e_fields, colorscale=colorscales[0], sizemode=sizemode, sizeref=e_sizeref, clim=e_clim, colorscales=colorscales)
        h_cones = PhotonicCrystal._fields_to_cones(h_fields, colorscale=colorscales[1], sizemode=sizemode, sizeref=h_sizeref, clim=h_clim, colorscales=colorscales) 


        for e_cone in e_cones:  
            e_fig.add_trace(e_cone)
        for h_cone in h_cones:
            h_fig.add_trace(h_cone)

        
        e_fig.update_layout(
            title=names[0],
            scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z')
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            showlegend=True
        )

        h_fig.update_layout(
            title=names[1],
            scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z')
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            showlegend=True
        )

        return e_fig, h_fig
        
        
    
    

    def plot_mode_fields_normal_to_k(self, mode, k):
        """
        Plot the fields perpendicular to the wavevector k for the mode.
        To be tested, not sure if works correctly.

        Args:
            mode (dict): The mode dictionary.
            k (numpy.ndarray): The wavevector [kx, ky, kz].

        Returns:
            tuple: A tuple containing the Plotly figures for the electric and magnetic fields.
        """
        fields = [mode["e_field"], mode["h_field"]]
        fields_norm_to_k = self._calculate_field_norm_to_k(fields, k)
        fig_e = go.Figure()
        fig_h = go.Figure()
        

        fig_e.add_trace(self._field_to_cones(fields_norm_to_k[0], colorscale="blues"))
        fig_h.add_trace(self._field_to_cones(fields_norm_to_k[1], colorscale="reds"))
        return fig_e, fig_h 
    
    def plot_vectorial_fields(self, fields, colorscales=["Viridis","Viridis"], names=["Field 1", "Field 2"]): 
        """
        Plot the vectorial fields of the modes.

        Args:
            fields (list): The list of fields to plot.
            colorscales (list, optional): The colorscales for the cones. Default is ["Viridis", "Viridis"].
            names (list, optional): The names of the fields. Default is ["Field 1", "Field 2"].

        Returns:
            tuple: A tuple containing the Plotly figures for the electric and magnetic fields.
        """

        fig_e = go.Figure()
        fig_h = go.Figure()
        fig_e.add_trace(self._field_to_cones(fields[0], colorscale=colorscales[0]))
        fig_h.add_trace(self._field_to_cones(fields[1], colorscale=colorscales[1]))
        fig_e.update_layout(
            title=names[0],
            scene=dict(
            xaxis=dict(title='Y'),
            yaxis=dict(title='X'),
            zaxis=dict(title='Z')
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            showlegend=True
        )

        fig_h.update_layout(
            title=names[1],
            scene=dict(
            xaxis=dict(title='Y'),
            yaxis=dict(title='X'),
            zaxis=dict(title='Z')
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            showlegend=True
        )
        return fig_e, fig_h

        
    
    
    @staticmethod
    def _field_to_cones(field, colorscale="Viridis", sizemode='absolute', sizeref=1, clim=(0, 1))-> go.Cone:
        """
        Convert a field to cones for visualization.
        Auxiliar method for plotting the fields.

        Args:
            field (np.ndarray): The field to convert.
            colorscale (str): The colorscale for the cones. Default is 'Viridis'.
            sizemode (str): The sizemode for the cones. Default is 'absolute'.
            sizeref (float): The sizeref for the cones. Default is 1.
            clim (tuple): The color limits for the cones. Default is (0, 1). (Not used now)

        Returns:
            go.Cone: The Plotly cone object for visualization.
        """
        field_x = np.real(field[..., 0])
        field_y = np.real(field[..., 1])
        field_z = np.real(field[..., 2])

        x, y, z = np.meshgrid(np.arange(field.shape[0]), np.arange(field.shape[1]), np.arange(field.shape[2]))


        cone = go.Cone(
            x=x.flatten(),
            y=y.flatten(),
            z=z.flatten(),
            u=field_x.flatten(),
            v=field_y.flatten(),
            w=field_z.flatten(),
            anchor='tail',
            sizemode=sizemode,
            sizeref=sizeref,
            colorscale=colorscale,
            #cmin = clim[0],
            #cmax = clim[1],
        )
        return cone

        
    @staticmethod
    def _fields_to_cones(fields, colorscale="Viridis", sizemode='absolute', sizeref=1, clim=(0, 1), colorscales=None)-> list:
        """
        Convert a list of fields to cones for visualization.
        Auxiliar method for plotting the fields.
        
        Args:
            fields (list): The list of fields to convert.
            colorscale (str): The colorscale for the cones. Default is 'Viridis'.
            sizemode (str): The sizemode for the
        - sizeref: The sizeref for the cones. Default is 1.
        - clim: The color limits for the cones. Default is (0, 1).
        - colorscales: The colorscales for the cones. Default is None.
        
        Returns:
        - cones: The list of cones for the fields.
        """

        
        cones = []

        if colorscales is None:
            colorscales = ["blues", "reds", "greens", "purples", "oranges", "ylorbr"]

        for i, field in enumerate( fields):
            cone = PhotonicCrystal._field_to_cones(field, colorscale=colorscales[i], sizemode=sizemode, sizeref=sizeref, clim=clim)
            cones.append(cone)
        return cones


    @staticmethod
    def _calculate_field_norm_to_k(fields, k):
        """
        Calculate the components of the field perpendicular to the wavevector k for each field in the list.

        Args:
            fields (list): A list of numpy arrays, each of shape (Nx, Ny, Nz, 3), where the last dimension contains the x, y, z components of the field.
            k (numpy.ndarray): A numpy array of shape (3,), representing the wavevector [kx, ky, kz].
        
        Returns:
            list: A list of numpy arrays, each of shape (Nx, Ny, Nz, 3), representing the field perpendicular to k for each input field.
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
    


    

    @staticmethod
    def _get_direction(k_vector):
        """
        Determine the primary direction of the wavevector k.

        Args:
            k_vector: A numpy array or list of shape (3,), representing the wavevector [kx, ky, kz].

        Returns:
            int: 0 for x-direction, 1 for y-direction, 2 for z-direction.
        """
        if k_vector[0] != 0 and k_vector[1] == 0 and k_vector[2] == 0:
            return 0  # x-direction
        elif k_vector[0] == 0 and k_vector[1] != 0 and k_vector[2] == 0:
            return 1  # y-direction
        elif k_vector[0] == 0 and k_vector[1] == 0 and k_vector[2] != 0:
            return 2  # z-direction
        else:
            raise ValueError("The wavevector k does not align with a primary axis.")
        

    
        
    @staticmethod
    def _calculate_effective_parameter(mode):
        """
        Calculate the effective parameters of the mode.

        Args:
            mode (dict): The mode dictionary.

        Returns:
            dict: A dictionary with the effective parameters of the mode.
        """
        raise NotImplementedError("calculate_effective_parameter method not implemented yet.")  

    def sweep_geometry_parameter(self, geom : partial, param_to_sweep: str, sweep_values: list, num_bands: int =4)-> list:
        
        """
        Sweep a parameter of the geometry and run simulations for each value.
        
        Args:
            geom (function): The geometry function to sweep.
            param_to_sweep (str): The parameter to sweep.
            sweep_values (list): The values to sweep.
            num_bands (int, optional): The number of bands to calculate. Defaults to 4.
        
        Returns:
            list: A list of dictionaries with the simulation data.

        """
        data = []
        old_geom  = self.geometry
        for value in sweep_values:
            kwargs = {param_to_sweep: value}
            self.geometry = geom(**kwargs)
            self.num_bands = num_bands
            self.set_solver(k_point=mp.Vector3())
            modes_zeven = self.run_simulation_with_output(runner="run_te", polarization="te")
            modes_zodd  = self.run_simulation_with_output(runner="run_tm", polarization="tm")
            data.append({
                'parameter_value': value,
                'modes_zeven': modes_zeven,
                'modes_zodd': modes_zodd,
                'parameter_name': param_to_sweep,
            })
        self.geometry = old_geom
        return data

    def plot_sweep_result(self, data, fig=None) -> go.Figure:
        """
        Plot the sweep result using Plotly.

        Args:
            data (list): The data from the sweep.
            fig (go.Figure): The Plotly figure to add the plot to.

        Returns:
            go.Figure: The Plotly figure object.
        """

        if fig is None:
            fig = go.Figure()

        # For each data parameter value in the x axis, add all the modes frequencies (mode["freq"])in the y axis
        num_bands = len(data[0]['modes_zeven'])
        for i in range(num_bands):
            modes_zeven = [d['modes_zeven'][i] for d in data]
            modes_zodd = [d['modes_zodd'][i] for d in data]
            param_values = [d['parameter_value'] for d in data]   
            fig.add_trace(go.Scatter(x=param_values, y=[m["freq"] for m in modes_zeven], mode='lines+markers', name=f'Band {i} TE', line=dict(color='red'), marker=dict(symbol=i, size=10)))
            fig.add_trace(go.Scatter(x=param_values, y=[m["freq"] for m in modes_zodd], mode='lines+markers', name=f'Band {i} TM', line=dict(color='blue'), marker=dict(symbol=i, size=10)))
        fig.update_layout(
            autosize=False,
            width=700,
            height=700,
        )   
        
        fig.update_layout(
            xaxis_title='Parameter Value',
            yaxis_title='Frequency (c/a)',
            title=f"Sweep of {data[0]['parameter_name']}",
            showlegend=True
        )

        return fig
    
    @staticmethod
    def basic_geometry():
        """ 
        Define the basic geometry of the photonic crystal.
        Must be implemented in the derived class.
        """

        raise NotImplementedError
    
    @staticmethod
    def basic_lattice():
        """
        Define the basic lattice of the photonic crystal.
        Must be implemented in the derived class.
        """

        raise NotImplementedError
  
    


@contextlib.contextmanager
def suppress_output():
    """
    Context manager to suppress stdout and stderr.

    This context manager redirects the standard output (stdout) and standard error (stderr)
    to os.devnull, effectively suppressing any output within its context.

    Yields:
        None: This context manager does not return any value.

    Example:
        >>> with suppress_output():
        ...     print("This will not be printed")
        ...     raise ValueError("This error will not be shown")
    """
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
    """
    This class provides methods to define and simulate 2D photonic crystals with various lattice types and geometries. 
    It allows for the calculation and visualization of dielectric distributions and electromagnetic fields within the crystal.
    
    Attributes:
        geometry_lattice (mp.Lattice): The lattice structure of the photonic crystal.
        k_points (list): List of k-points for the simulation.
        geometry (list): List of geometric objects defining the photonic crystal.
        k_points_interpolated (list): Interpolated k-points for the simulation.
        epsilon (ndarray): Dielectric distribution of the photonic crystal.
    
    Methods:
        __init__(self, lattice_type=None, num_bands=6, resolution=(32, 32), interp=4, periods=3, pickle_id=None, geometry=None, use_XY=True, k_point_max=0.2):
            Initializes the Crystal2D object with the specified parameters.
        plot_epsilon(self, fig=None, title='Epsilon', **kwargs):
            Plots the dielectric distribution interactively using Plotly.
        plot_field(self, target_polarization, target_k_point, target_frequency, frequency_tolerance=0.01, k_point_max_distance=None, periods=5, component=2, quantity="real", colorscale='RdBu'):
            Plots the electromagnetic field distribution interactively using Plotly.
        plot_field_components(self, target_polarization, target_k_point, target_frequency, frequency_tolerance=0.01, k_point_max_distance=None, periods=1, quantity="real", colorscale='RdBu'):
            Plots the field components (Ex, Ey, Ez) and (Hx, Hy, Hz) for specific modes with consistent color scales.
        basic_geometry(radius_1=0.2, eps_atom_1=1, radius_2=None, eps_atom_2=None, eps_bulk=12):
            Defines a basic geometry for the photonic crystal.
        ellipsoid_geometry(e1=0.2, e2=0.3, eps_atom=1, eps_bulk=12):
            Defines an ellipsoid geometry for the photonic crystal.
        advanced_material_geometry(radius_1=0.2, epsilon_diag=mp.Vector3(12, 12, 12), epsilon_offdiag=mp.Vector3(0, 0, 0), chi2_diag=mp.Vector3(0, 0, 0), chi3_diag=mp.Vector3(0, 0, 0), eps_atom_1=1):
            Defines an advanced material geometry for the photonic crystal.
        basic_lattice(lattice_type='square'):
            Defines the basic lattice structure for the photonic crystal.
        square_lattice():
            Defines a square lattice for the photonic crystal.
        triangular_lattice():
            Defines a triangular lattice for the photonic crystal.
    """
    

    
    def __init__(self,
                lattice_type = "square",
                num_bands: int = 6,
                resolution: tuple[int, int] | int = 32,
                interp: int =4,
                periods: int =3, 
                pickle_id = None,
                geometry = None,
                use_XY = True,
                k_point_max = 0.2):
        
        """
        Initializes the Crystal2D object with the specified parameters.

        Args:
            lattice_type (str): The type of lattice. Default is 'square' other option is 'triangular'. It determines the k-points if use_XY is False.
            num_bands (int): The number of bands to calculate. Default is 6.
            resolution (tuple[int, int] | int): The resolution of the simulation. Default is (32, 32).
            interp (int): The interpolation factor for k-points. Default is 4.
            periods (int): The number of periods to simulate. Default is 3.
            pickle_id (str): The ID for pickling the simulation. Default is None.
            geometry (list): The geometry of the photonic crystal. Default is None.
            use_XY (bool): Whether to use the X and Y directions for the x-axis or high symmetry points. Default is True.
        """
      
        super().__init__(lattice_type, num_bands, resolution, interp, periods, pickle_id, use_XY=use_XY)
        
        self.geometry_lattice, self.k_points = self.basic_lattice(lattice_type)
        if use_XY is True:
            self.k_points = [
                mp.Vector3(k_point_max, 0, 0),      # X
                mp.Vector3(0, 0 ,0 ),       # Gamma
                mp.Vector3(0, k_point_max,0)        # Y
            ]
        self.geometry = geometry if geometry is not None else self.basic_geometry()
        self.k_points_interpolated = mp.interpolate(interp, self.k_points)
        


    def plot_epsilon(self, fig=None, title='Epsilon', **kwargs)-> go.Figure:
        """
        Plot the dielectric distribution interactively using Plotly.

        Args:
            fig (go.Figure, optional): The Plotly figure to add the epsilon plot to. Default is None.
            title (str, optional): The title of the plot. Default is 'Epsilon'.
            **kwargs: Additional keyword arguments for Plotly.

        Returns:
            go.Figure: The Plotly figure object.
        """

        with suppress_output():
            if self.epsilon is None:
                md = mpb.MPBData(rectify=True, periods=self.periods, resolution=self.resolution)
                converted_eps = md.convert(self.ms.get_epsilon())
            else:
                converted_eps = self.epsilon
            if fig is None:
                fig = go.Figure()

            fig.add_trace(go.Heatmap(z=converted_eps.T, colorscale='Viridis'))
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
        
    def plot_field(self,    
                target_polarization,
                target_k_point,
                target_frequency,
                frequency_tolerance=0.01,
                k_point_max_distance=None,
                periods=5,
                component: int = 2,
                quantity: str = "real",
                colorscale: str = "RdBu",
    )-> tuple[go.Figure, go.Figure]:
        """
        Plot the electromagnetic field distribution using Plotly.

        Args:
            target_polarization (str): The polarization of the mode.
            target_k_point (tuple): The k-point of the mode.
            target_frequency (float): The frequency of the mode.
            frequency_tolerance (float): The tolerance for frequency similarity.
            k_point_max_distance (float, optional): The maximum distance for k-point similarity. Default is None.
            periods (int): The number of periods to extract. Default is 5.
            component (int): The component of the field to plot. x=0, y=1, z=2. Default is 2.
            quantity (str): The quantity to plot. Default is 'real'.
            colorscale (str): The colorscale for the heatmap. Default is 'RdBu'.

        Returns:
            tuple[go.Figure, go.Figure]: The Plotly figure objects for the electric and magnetic fields.
        """

        fig_e = go.Figure()
        fig_h = go.Figure()
        target_modes = self.look_for_mode(target_polarization, target_k_point, target_frequency, frequency_tolerance, k_point_max_distance)
        if not target_modes:
            print("No modes found with the specified criteria.")
            return

        with suppress_output():
            self.run_dumb_simulation()
            md = mpb.MPBData(rectify=True, periods=periods, lattice=self.ms.get_lattice())
            eps = md.convert(self.ms.get_epsilon()) 

        min_eps, max_eps = np.min(eps), np.max(eps)
        midpoint = (max_eps + min_eps) / 2
        num_plots = len(target_modes)
        dropdown_buttons_e = []
        dropdown_buttons_h = []

        for i, mode in enumerate(target_modes):
            # Initialize visibility status: False for all traces
            visible_status_e = [False] * (2 * num_plots)  # Each mode adds two traces
            visible_status_h = [False] * (2 * num_plots)

            visible_status_e[2 * i] = True  # Set the contour plot visible
            visible_status_h[2 * i] = True
            visible_status_e[2 * i + 1] = True  # Set the heatmap visible
            visible_status_h[2 * i + 1] = True  # Set the heatmap visible

            k_point = mode["k_point"]
            freq    = mode["freq"]
            polarization = mode["polarization"]

            # Take the specified component of the fields in the center of the slab
            
            e_field = mpb.MPBArray(mode["e_field"], lattice = self.ms.get_lattice(),  kpoint = mode["k_point"] )
            h_field = mpb.MPBArray(mode["h_field"], lattice = self.ms.get_lattice(),  kpoint = mode["k_point"])
            e_field = e_field[..., component]
            h_field = h_field[..., component]
            e_field = np.squeeze(e_field)
            h_field = np.squeeze(h_field)   

            with suppress_output():
                e_field = md.convert(e_field) 
                h_field = md.convert(h_field)

            if quantity == "real":
                e_field = np.real(e_field)
                h_field = np.real(h_field)
            elif quantity == "imag":
                e_field = np.imag(e_field)
                h_field = np.imag(h_field)
            elif quantity == "abs":
                e_field = np.abs(e_field)
                h_field = np.abs(h_field)
            else:
                raise ValueError("Invalid quantity. Choose 'real', 'imag', or 'abs'.")

            # Add the contour plot for permittivity (eps) at the midpoint
            contour_e = go.Contour(z=eps.T,
                                contours=dict(
                                    start=midpoint,  # Start and end at the midpoint to ensure a single level
                                    end=midpoint,
                                    size=0.1,  # A small size to keep it as a single contour
                                    coloring='none'  # No filling
                                ),
                                line=dict(color='black', width=2),
                                showscale=False,
                                opacity=0.7,
                                visible=True if i ==  0  else False)  # Only the first mode is visible
            contour_h = contour_e  # Same contour for H-field figure

            # Add the contour trace
            fig_e.add_trace(contour_e)
            fig_h.add_trace(contour_h)

            # Add the heatmap for the electric field
            heatmap_e = go.Heatmap(z=e_field.T, colorscale=colorscale, zsmooth='best', opacity=0.9,
                                    showscale=True, visible= True if i == 0 else False)
            # Add the heatmap for the magnetic field
            heatmap_h = go.Heatmap(z=h_field.T, colorscale=colorscale, zsmooth='best', opacity=0.9,
                                    showscale=True, visible= True if i == 0 else False)

            # Add the heatmap trace
            fig_e.add_trace(heatmap_e)
            fig_h.add_trace(heatmap_h)
            
            data_str = f"Mode {i + 1} <br> k = [{k_point[0]:0.2f}, {k_point[1]:0.2f}], freq={freq:0.3f}, polarization={polarization}"
            if component == 0: 
                component_str = "x-component"
            elif component == 1:
                component_str = "y-component"
            else:
                component_str = "z-component"
            subtitle_e = f"E-field, {component_str}, {quantity}"
            subtitle_h = f"H-field, {component_str}, {quantity}"
            # Create a button for each field dataset for the dropdown
            dropdown_buttons_e.append(dict(label=f'Mode {i + 1}',
                                        method='update',
                                        args=[{'visible': visible_status_e},  # Update visibility for both eps and field
                                            {'title':f"{data_str}:<br> {subtitle_e}"}
                                        ]))
            dropdown_buttons_h.append(dict(label=f'Mode {i + 1}',
                                        method='update',
                                        args=[{'visible': visible_status_h},  # Update visibility for both eps and field
                                            {'title':f"{data_str}:<br> {subtitle_h}"}
                                        ]))
        k_point = target_modes[0]["k_point"]
        freq    = target_modes[0]["freq"]
        data_str = f"Mode {0} <br> k = [{k_point[0]:0.2f}, {k_point[1]:0.2f}], freq={freq:0.3f}, polarization={polarization}"
        
        fig_e.update_layout(
            updatemenus=[dict(active=0,
                            buttons=dropdown_buttons_e,
                            x=1.15, y=1.15,
                            xanchor='left', yanchor='top')],
            title=f"{data_str}:<br> {subtitle_e}",
            xaxis_showgrid=False, yaxis_showgrid=False,
            xaxis_zeroline=False, yaxis_zeroline=False,
            xaxis_visible=False, yaxis_visible=False,
            hovermode="closest",
            width=800, height=800,
            xaxis_title="X",
            yaxis_title="Y"
            
        )

        fig_h.update_layout(
            updatemenus=[dict(active=0,
                            buttons=dropdown_buttons_h,
                            x=1.15, y=1.15,
                            xanchor='left', yanchor='top')],
            title=f"{data_str}:<br> {subtitle_h}",
            xaxis_showgrid=False, yaxis_showgrid=False,
            xaxis_zeroline=False, yaxis_zeroline=False,
            xaxis_visible=False, yaxis_visible=False,
            hovermode="closest",
            width=800, height=800,
            xaxis_title="X",
            yaxis_title="Y"
        )

        return fig_e, fig_h
    

    def plot_field_components(self,
                            target_polarization,
                            target_k_point,
                            target_frequency,
                            frequency_tolerance=0.01,
                            k_point_max_distance=None,
                            periods: int = 1,
                            quantity: str = "real",
                            colorscale: str = 'RdBu',
                            )-> tuple:
        """
        Plot the field components (Ex, Ey, Ez) and (Hx, Hy, Hz) for specific modes with consistent color scales.

        Args:
            target_polarization (str): The polarization of the target mode.
            target_k_point (tuple): The k-point of the target mode.
            target_frequency (float): The frequency of the target mode.
            frequency_tolerance (float): The tolerance for frequency similarity.
            periods (int): The number of periods to extract. Default is 1.
            quantity (str): The quantity to plot ('real', 'imag', or 'abs'). Default is 'real'.
            colorscale (str): The colorscale to use for the plot. Default is 'RdBu'.

        Returns:
            tuple: A tuple containing the electric field figure and the magnetic field figure.
        """

        target_modes = self.look_for_mode(target_polarization, target_k_point, target_frequency,
                                        freq_tolerance=frequency_tolerance, k_point_max_distance=k_point_max_distance)
        print(f"Number of target modes found: {len(target_modes)}")

        with suppress_output():
            self.run_dumb_simulation()
            md = mpb.MPBData(rectify=True, periods=periods, lattice=self.ms.get_lattice())
            eps = md.convert(self.ms.get_epsilon())

        # Calculate the midpoint between min and max of the permittivity (eps)
        min_eps, max_eps = np.min(eps), np.max(eps)
        midpoint = (min_eps + max_eps) / 2  # The level to be plotted

        fig_e = make_subplots(rows=1, cols=3, subplot_titles=("Ex", "Ey", "Ez"))
        fig_h = make_subplots(rows=1, cols=3, subplot_titles=("Hx", "Hy", "Hz"))
        dropdown_buttons_e = []
        dropdown_buttons_h = []

        # For each mode, calculate separate min and max values for Ex, Ey, Ez (and similarly Hx, Hy, Hz)
        for i, mode in enumerate(target_modes):
            # Get field arrays for this mode
            e_field_array = mpb.MPBArray(mode["e_field"], lattice=self.ms.get_lattice(), kpoint=mode["k_point"])
            h_field_array = mpb.MPBArray(mode["h_field"], lattice=self.ms.get_lattice(), kpoint=mode["k_point"])

            # Extract field components in the center of the slab
            e_field_x = e_field_array[..., 0]  # Shape (Nx, Ny)
            e_field_y = e_field_array[..., 1]  # Shape (Nx, Ny)
            e_field_z = e_field_array[..., 2]  # Shape (Nx, Ny)
            h_field_x = h_field_array[..., 0]  # Shape (Nx, Ny)
            h_field_y = h_field_array[..., 1]  # Shape (Nx, Ny)
            h_field_z = h_field_array[..., 2]  # Shape (Nx, Ny)

            # Convert the field arrays to the correct shape
            e_field_x = np.squeeze(e_field_x)
            e_field_y = np.squeeze(e_field_y)
            e_field_z = np.squeeze(e_field_z)
            h_field_x = np.squeeze(h_field_x)
            h_field_y = np.squeeze(h_field_y)
            h_field_z = np.squeeze(h_field_z)
            

            with suppress_output():
                e_field_x = md.convert(e_field_x)
                e_field_y = md.convert(e_field_y)
                e_field_z = md.convert(e_field_z)
                h_field_x = md.convert(h_field_x)
                h_field_y = md.convert(h_field_y)
                h_field_z = md.convert(h_field_z)

            e_field = np.stack([e_field_x, e_field_y, e_field_z], axis=-1)
            h_field = np.stack([h_field_x, h_field_y, h_field_z], axis=-1)                                    

            # Select quantity to display (real, imag, abs)
            if quantity == "real":
                e_field = np.real(e_field)
                h_field = np.real(h_field)
            elif quantity == "imag":
                e_field = np.imag(e_field)
                h_field = np.imag(h_field)
            elif quantity == "abs":
                e_field = np.abs(e_field)
                h_field = np.abs(h_field)
            else:
                raise ValueError("Invalid quantity. Choose 'real', 'imag', or 'abs'.")

            # Calculate the component-specific min/max for E and H fields of this mode
            e_min = np.min(e_field)
            e_max = np.max(e_field)
            h_min = np.min(h_field)
            h_max = np.max(h_field)

            # Components of the E and H fields
            Ex, Ey, Ez = e_field[..., 0], e_field[..., 1], e_field[..., 2]
            Hx, Hy, Hz = h_field[..., 0], h_field[..., 1], h_field[..., 2]

            # Define visibility settings per mode, including contours as always visible
            visible_status_e = [False] * (len(target_modes) * 6)  # 3 components per mode, with contour for each
            visible_status_h = [False] * (len(target_modes) * 6)
            # Make the contour visible by default


            # Make this mode's components and the corresponding contour visible in the initial layout 
            for j in range(6):
                visible_status_e[6*i + j] = True
                visible_status_h[6*i + j] = True


            # Add contour traces for permittivity to each subplot of fig_e and fig_h for this mode
            fig_e.add_trace(go.Contour(z=eps.T, contours=dict(start=midpoint, end=midpoint, size=0.1, coloring='none'), line=dict(color='black', width=2), showscale=False, opacity=0.7, showlegend=False, visible=True if i == len(target_modes)-1 else False), row=1, col=1)
            fig_e.add_trace(go.Contour(z=eps.T, contours=dict(start=midpoint, end=midpoint, size=0.1, coloring='none'), line=dict(color='black', width=2), showscale=False, opacity=0.7, showlegend=False, visible=True if i == len(target_modes)-1 else False), row=1, col=2)
            fig_e.add_trace(go.Contour(z=eps.T, contours=dict(start=midpoint, end=midpoint, size=0.1, coloring='none'), line=dict(color='black', width=2), showscale=False, opacity=0.7, showlegend=False, visible=True if i == len(target_modes)-1 else False), row=1, col=3)

            fig_h.add_trace(go.Contour(z=eps.T, contours=dict(start=midpoint, end=midpoint, size=0.1, coloring='none'), line=dict(color='black', width=2), showscale=False, opacity=0.7, showlegend=False, visible=True if i == len(target_modes)-1 else False), row=1, col=1)
            fig_h.add_trace(go.Contour(z=eps.T, contours=dict(start=midpoint, end=midpoint, size=0.1, coloring='none'), line=dict(color='black', width=2), showscale=False, opacity=0.7, showlegend=False, visible=True if i == len(target_modes)-1 else False), row=1, col=2)
            fig_h.add_trace(go.Contour(z=eps.T, contours=dict(start=midpoint, end=midpoint, size=0.1, coloring='none'), line=dict(color='black', width=2), showscale=False, opacity=0.7, showlegend=False, visible=True if i == len(target_modes)-1 else False), row=1, col=3)

            # Add Ex, Ey, Ez with shared colorbar limits for the E field of this mode
            fig_e.add_trace(go.Heatmap(z=Ex.T, colorscale=colorscale, showscale=False, zmin=e_min, zmax=e_max, visible=True if i == len(target_modes)-1 else False, zsmooth="best", opacity=0.8), row=1, col=1)
            fig_e.add_trace(go.Heatmap(z=Ey.T, colorscale=colorscale, showscale=False, zmin=e_min, zmax=e_max, visible=True if i == len(target_modes)-1 else False, zsmooth="best", opacity=0.8), row=1, col=2)
            fig_e.add_trace(go.Heatmap(z=Ez.T, colorscale=colorscale, showscale=True, colorbar=dict(title="E-field", len=0.75, thickness=15), zmin=e_min, zmax=e_max, visible=True if i == len(target_modes)-1 else False, zsmooth="best", opacity=0.8), row=1, col=3)

            # Add Hx, Hy, Hz with shared colorbar limits for the H field of this mode
            fig_h.add_trace(go.Heatmap(z=Hx.T, colorscale=colorscale, showscale=False, zmin=h_min, zmax=h_max, visible=True if i == len(target_modes)-1 else False, zsmooth="best", opacity=0.8), row=1, col=1)
            fig_h.add_trace(go.Heatmap(z=Hy.T, colorscale=colorscale, showscale=False, zmin=h_min, zmax=h_max, visible=True if i == len(target_modes)-1 else False, zsmooth="best", opacity=0.8), row=1, col=2)
            fig_h.add_trace(go.Heatmap(z=Hz.T, colorscale=colorscale, showscale=True, colorbar=dict(title="H-field", len=0.75, thickness=15), zmin=h_min, zmax=h_max, visible=True if i == len(target_modes)-1 else False, zsmooth="best", opacity=0.8), row=1, col=3)
            
            

            # Dropdown data for E-field
            k_point = mode["k_point"]
            freq = mode["freq"]
            polarization = mode["polarization"]
            mode_description = f"Mode {i + 1}<br>k = [{k_point[0]:0.2f}, {k_point[1]:0.2f}], freq={freq:0.3f}, polarization={polarization}"
            
            dropdown_buttons_e.append(
                dict(label=f"Mode {i + 1}",
                    method='update',
                    args=[{'visible': visible_status_e},
                        {'title': f"{mode_description}: {quantity} of E-field components"}]))

            dropdown_buttons_h.append(
                dict(label=f"Mode {i + 1}",
                    method='update',
                    args=[{'visible': visible_status_h},
                        {'title': f"{mode_description}: {quantity} of H-field components"}]))

        # Layout and color settings
        fig_e.update_layout(
            title=f"{mode_description}: {quantity} of E-field components",
            updatemenus=[dict(
                active=len(target_modes) - 1,
                buttons=dropdown_buttons_e)],
            coloraxis=dict(colorbar=dict(len=0.75)),
            width=1200, height=400,
            xaxis_showgrid=False, yaxis_showgrid=False,
            xaxis_zeroline=False, yaxis_zeroline=False,
            hovermode="closest"
        )

        fig_h.update_layout(
            title=f"{mode_description}: {quantity} of H-field components",
            updatemenus=[dict(
                active=len(target_modes) - 1,
                buttons=dropdown_buttons_h)],
            coloraxis=dict(colorbar=dict(len=0.75)),
            width=1200, height=400,
            xaxis_showgrid=False, yaxis_showgrid=False,
            xaxis_zeroline=False, yaxis_zeroline=False,
            hovermode="closest"
        )

        # Final adjustments
        fig_e.update_xaxes(showticklabels=False)
        fig_e.update_yaxes(showticklabels=False)
        fig_h.update_xaxes(showticklabels=False)
        fig_h.update_yaxes(showticklabels=False)

        fig_e.update_layout(yaxis_scaleanchor="x", yaxis_scaleratio=1)
        fig_h.update_layout(yaxis_scaleanchor="x", yaxis_scaleratio=1)
        fig_e.update_layout(yaxis2_scaleanchor="x2", yaxis2_scaleratio=1)
        fig_h.update_layout(yaxis2_scaleanchor="x2", yaxis2_scaleratio=1)
        fig_e.update_layout(yaxis3_scaleanchor="x3", yaxis3_scaleratio=1)
        fig_h.update_layout(yaxis3_scaleanchor="x3", yaxis3_scaleratio=1)


        fig_e.update_xaxes(title_text="X-axis", row=1, col=1)
        fig_e.update_yaxes(title_text="Y-axis", row=1, col=1)
        fig_e.update_xaxes(title_text="X-axis", row=1, col=2)
        fig_e.update_yaxes(title_text="Y-axis", row=1, col=2)
        fig_e.update_xaxes(title_text="X-axis", row=1, col=3)
        fig_e.update_yaxes(title_text="Y-axis", row=1, col=3)

        fig_h.update_xaxes(title_text="X-axis", row=1, col=1)
        fig_h.update_yaxes(title_text="Y-axis", row=1, col=1)
        fig_h.update_xaxes(title_text="X-axis", row=1, col=2)
        fig_h.update_yaxes(title_text="Y-axis", row=1, col=2)
        fig_h.update_xaxes(title_text="X-axis", row=1, col=3)
        fig_h.update_yaxes(title_text="Y-axis", row=1, col=3)

        return fig_e, fig_h
        

    
    
    
    

    
    
    @staticmethod
    def basic_lattice(lattice_type='square')-> tuple[mp.Lattice, list[mp.Vector3]]:
        """
        Define the basic lattice of the photonic crystal.

        Args:
            lattice_type (str): The type of lattice. Default is 'square'.

        Returns:
            tuple: A tuple containing the lattice object representing the lattice and a list of k-points for the simulation.
        """
        if lattice_type == 'square':
            return Crystal2D.square_lattice()
        elif lattice_type == 'triangular':
            return Crystal2D.triangular_lattice()
        else:
            raise ValueError("Invalid lattice type. Choose 'square' or 'triangular'.")
        


    @staticmethod
    def square_lattice()-> tuple[mp.Lattice, list[mp.Vector3]]:
        """
        Define the square lattice for the photonic crystal.

        Returns:
            tuple: A tuple containing the lattice object representing the square lattice and a list of k-points for the simulation.
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
    def triangular_lattice()-> tuple[mp.Lattice, list[mp.Vector3]]:
        """
        Define the triangular lattice for the photonic crystal.

        Returns:
            tuple: A tuple containing the lattice object representing the triangular lattice and a list of k-points for the simulation.
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
    

    @staticmethod
    def basic_geometry(radius_1=0.2, 
                       eps_atom_1=1, 
                       radius_2=None, 
                       eps_atom_2=None,
                       eps_bulk = 12, 
                       )-> list: 
        """
        Define a basic geometry for the photonic crystal.
        So far only one atom has been tested. It uses cylinders to define the geometry.

        Args:
            radius_1 (float): The radius of the first atom. Default is 0.2.
            eps_atom_1 (float): The dielectric constant of the first atom. Default is 1.
            radius_2 (float, optional): The radius of the second atom. Default is None.
            eps_atom_2 (float, optional): The dielectric constant of the second atom. Default is None.
            eps_bulk (float): The dielectric constant of the bulk. Default is 12.

        Returns:
            list: A list of geometric objects defining the photonic crystal.

        
        """

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
    def ellipsoid_geometry(e1: float=0.2, 
                           e2: float = 0.3,
                           eps_atom: float = 1, 
                           eps_bulk: float= 12)->list:
        """
        Define an ellipsoid geometry for the photonic crystal.

        Args:
            e1 (float): The first radius of the ellipsoid. Default is 0.2.
            e2 (float): The second radius of the ellipsoid. Default is 0.3.
            eps_atom (float): The dielectric constant of the atom. Default is 1.
            eps_bulk (float): The dielectric constant of the bulk. Default is 12.

        Return
            list: A list of geometric objects defining the photonic crystal.
        """

        geometry = [
            mp.Block(
                size = mp.Vector3(mp.inf, mp.inf),
                material=mp.Medium(epsilon=eps_bulk)),
            ]   
        
        size=mp.Vector3(e1,e2, mp.inf)
        geometry.append(mp.Ellipsoid(size=size,
                                     material=mp.Medium(epsilon=eps_atom),
                                     center=mp.Vector3(0,0)))
        return geometry
    
    @staticmethod
    def advanced_material_geometry(
        radius_1 = 0.2,
        epsilon_diag = mp.Vector3(12, 12, 12),
        epsilon_offdiag = mp.Vector3(0, 0, 0),
        chi2_diag = mp.Vector3(0,0,0),
        
        chi3_diag = mp.Vector3(0,0,0),
        
        eps_atom_1 = 1
    )-> list:
        """
        Define an advanced material geometry for the photonic crystal.

        Args:
            radius_1 (float): The radius of the atom. Default is 0.2.
            epsilon_diag (mp.Vector3): The diagonal components of the dielectric tensor. Default is (12, 12, 12).
            epsilon_offdiag (mp.Vector3): The off-diagonal components of the dielectric tensor. Default is (0, 0, 0).
            chi2_diag (mp.Vector3): The diagonal components of the second-order nonlinear susceptibility tensor. Default is (0, 0, 0).
            chi3_diag (mp.Vector3): The diagonal components of the third-order nonlinear susceptibility tensor. Default is (0, 0, 0).
            eps_atom_1 (float): The dielectric constant of the atom. Default is 1.

        Returns:
            list: A list of geometric objects defining the photonic crystal.
        """

        
        geometry =[
            mp.Block(
                size = mp.Vector3(mp.inf, mp.inf),
                material = mp.Medium(
                    epsilon_diag=epsilon_diag,
                    epsilon_offdiag = epsilon_offdiag, 
                    E_chi2_diag = chi2_diag, 
                    E_chi3_diag = chi3_diag,
                )
            )
        ]

        geometry.append(mp.Cylinder(radius_1, 
                                        material=mp.Medium(epsilon=eps_atom_1),
                                        center = mp.Vector3(0,0)))
        return geometry
    
       
    
    


class CrystalSlab(PhotonicCrystal):
    """
    CrystalSlab class represents a photonic crystal slab structure, inheriting from the PhotonicCrystal class.
    This class provides methods to define the lattice, geometry, and to plot various properties of the photonic crystal slab.
    Geometries can be added using mpb functions such as Cylinder, Ellipsoid, etc.


    Attributes:
        geometry_lattice (mp.Lattice): The lattice structure of the photonic crystal.
        k_points (list): List of k-points for the simulation.
        geometry (list): List of geometric objects defining the photonic crystal.
        k_points_interpolated (list): Interpolated k-points for the simulation.
        epsilon (np.ndarray): Dielectric function values obtained from the simulation.

    Methods:
        __init__(self, lattice_type=None, num_bands=4, resolution=mp.Vector3(32,32,16), interp=2, periods=3, pickle_id=None, geometry=None, use_XY=True, k_point_max=0.2):
            Initializes the CrystalSlab object with the given parameters.
        plot_epsilon(self, fig=None, opacity=0.3, colorscale='PuBuGn', override_resolution_with=None, periods=1):
            Plots the epsilon values obtained from the simulation using Plotly.
        basic_lattice(lattice_type='square', height_supercell=4):
            Defines the basic lattice structure for the photonic crystal.
        square_lattice(height_supercell=4):
            Defines the square lattice for the photonic crystal.
        triangular_lattice(height_supercell=4):
            Defines the triangular lattice for the photonic crystal.
        basic_geometry(radius_1=0.2, eps_atom_1=1, radius_2=None, eps_atom_2=None, eps_bulk=12, height_supercell=4, height_slab=0.5, eps_background=1, eps_substrate=None):
            Defines the basic geometry for the photonic crystal.
        ellipsoid_geometry(e1=0.2, e2=0.3, eps_atom=1, height_supercell=4, height_slab=0.5, eps_background=1, eps_substrate=1, eps_diag=mp.Vector3(12, 12, 12), eps_offdiag=mp.Vector3(0, 0, 0), E_chi2_diag=mp.Vector3(0,0,0), E_chi3_diag=mp.Vector3(0,0,0)):
            Defines the geometry with ellipsoidal atoms for the photonic crystal.
        advanced_material_geometry(radius_1=0.2, epsilon_diag=mp.Vector3(12, 12, 12), epsilon_offdiag=mp.Vector3(0, 0, 0), chi2_diag=mp.Vector3(0,0,0), chi3_diag=mp.Vector3(0,0,0), eps_atom_1=1, eps_background=1, eps_substrate=1, height_supercell=4, height_slab=0.5):
            Defines the advanced material geometry for the photonic crystal.
        plot_field(self, target_polarization, target_k_point, target_frequency, frequency_tolerance=0.01, k_point_max_distance=None, periods=1, component=2, quantity='real', colorscale='RdBu'):
            Plots the field for a specific mode based on the given parameters.
        plot_field_components(self, target_polarization, target_k_point, target_frequency, frequency_tolerance=0.01, k_point_max_distance=None, periods=1, quantity='real', colorscale='RdBu'):
            Plots the field components (Ex, Ey, Ez) and (Hx, Hy, Hz) for specific modes with consistent color scales.
    """

    def __init__(self,
                lattice_type = "square",
                num_bands: int = 4,
                resolution = mp.Vector3(32,32,16),
                interp: int =2,
                periods: int =3, 
                pickle_id = None,
                geometry = None, 
                use_XY = True,
                k_point_max = 0.2):
        """
        Initializes the CrystalSlab object with the given parameters.

        Args:
            lattice_type (str): The type of lattice. It can be 'square' or 'triangular'. Default is 'square'.
            num_bands (int): The number of bands to calculate. Default is 4.
            resolution (mp.Vector3): The resolution of the simulation. Default is mp.Vector3(32, 32, 16).
            interp (int): The interpolation factor for k-points. Default is 2.
            periods (int): The number of periods to use in some plotting functions. Default is 3.
            pickle_id (str): The ID for pickling the simulation. Default is None.
            geometry (list): The geometry of the photonic crystal. Default is None. If it is none, the basic geometry is used.
            use_XY (bool): Whether to use the X and Y directions for the x-axis or high symmetry points. Default is True.
            k_point_max (float): The maximum k-point value. Default is 0.2.
        """


        super().__init__(lattice_type, num_bands, resolution, interp, periods, pickle_id, use_XY=True)
        
        
        self.geometry_lattice, self.k_points = self.basic_lattice(lattice_type)
        if use_XY is True: 
            self.k_points = [
                mp.Vector3(k_point_max, 0, 0),      # X
                mp.Vector3(0, 0 ,0 ),               # Gamma
                mp.Vector3(0, k_point_max, 0)       # Y
            ]
        self.geometry = geometry if geometry is not None else self.basic_geometry()
        self.k_points_interpolated = mp.interpolate(interp, self.k_points)

    def plot_epsilon(self,
                    fig=None, 
                    opacity=0.3, 
                    colorscale='PuBuGn', 
                    override_resolution_with: None|int= None, 
                    periods = 1,
                    )-> go.Figure:
        """
        Plot the epsilon values obtained from the simulation using Plotly.

        Args:
            fig (go.Figure, optional): The Plotly figure to add the epsilon plot to. Default is None.
            opacity (float, optional): The opacity of the plot. Default is 0.3.
            colorscale (str, optional): The colorscale for the plot. Default is 'PuBuGn'.
            override_resolution_with (int, optional): The resolution to use for plotting. You can change the resolution just for the plot, but the simulation will still use the value set in the __init__ method. Default is None.
            periods (int, optional): The number of periods to plot. Default is 1.

        Returns:
            go.Figure: The Plotly figure object.
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
        print(epsilon.shape)
        epsilon = np.transpose(epsilon,(1,0,2)) 

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
            xaxis=dict(title='X', visible=True),
            yaxis=dict(title='Y', visible=True),
            zaxis=dict(title='Z', visible=True),
            )
        )
        
        fig.update_layout(height=800, width=600)

        return fig

    @staticmethod
    def basic_lattice(lattice_type='square', height_supercell=4)-> tuple[mp.Lattice, list[mp.Vector3]]:
        """
        Define the basic lattice structure for the photonic crystal.

        Args:
            lattice_type (str): The type of lattice. Default is 'square'.
            height_supercell (int): The height of the supercell. Default is 4.

        Returns:
            tuple: A tuple containing the lattice object representing the lattice and a list of k-points for the simulation.
        """

        if lattice_type == 'square':
            return CrystalSlab.square_lattice()
        elif lattice_type == 'triangular':
            return CrystalSlab.triangular_lattice()
        else:
            raise ValueError("Invalid lattice type. Choose 'square' or 'triangular'.")
        
    @staticmethod
    def square_lattice(height_supercell=4)-> tuple[mp.Lattice, list[mp.Vector3]]:
        """
        Define the square lattice for the photonic crystal.

        Returns:
            tuple: A tuple containing the lattice object representing the square lattice and a list of k-points for the simulation.
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
    def triangular_lattice(height_supercell=4)-> tuple[mp.Lattice, list[mp.Vector3]]:
        """
        Define the triangular lattice for the photonic crystal.

        Returns:
            tuple: A tuple containing the lattice object representing the triangular lattice and a list of k-points for the simulation.
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
                       )-> list:
        """
        Define the basic geometry for the photonic crystal.
        Biatomic structures have not been tested. 
        
        Args:
            radius_1 (float): The radius of the first atom. Default is 0.2.
            eps_atom_1 (float): The dielectric constant of the first atom. Default is 1.
            radius_2 (float, optional): The radius of the second atom. Default is None.
            eps_atom_2 (float, optional): The dielectric constant of the second atom. Default is None.
            eps_bulk (float): The dielectric constant of the bulk. Default is 12.
            height_supercell (float): The height of the supercell. Default is 4.
            height_slab (float): The height of the slab. Default is 0.5.
            eps_background (float): The dielectric constant of the background. Default is 1.
            eps_substrate (float, optional): The dielectric constant of the substrate. Default is None.

        Returns:
            list: A list of geometric objects defining the photonic crystal.
        """
        
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
    

    def ellipsoid_geometry(e1: float=0.2,
                           e2: float = 0.3,
                           eps_atom = 1,
                           height_supercell=4,
                           height_slab=0.5,
                           eps_background=1,
                           eps_substrate=1,
                           eps_diag = mp.Vector3(12, 12, 12),
                           eps_offdiag = mp.Vector3(0, 0, 0),
                           E_chi2_diag = mp.Vector3(0,0,0),
                           E_chi3_diag = mp.Vector3(0,0,0),
                           )-> list:       
        """
        Define an ellipsoid geometry for the photonic crystal.

        Args:
            e1 (float): The first radius of the ellipsoid. Default is 0.2.
            e2 (float): The second radius of the ellipsoid. Default is 0.3.
            eps_atom (float): The dielectric constant of the atom. Default is 1.
            height_supercell (float): The height of the supercell. Default is 4.
            height_slab (float): The height of the slab. Default is 0.5.
            eps_background (float): The dielectric constant of the background. Default is 1.
            eps_substrate (float): The dielectric constant of the substrate. Default is 1.
            eps_diag (mp.Vector3): The diagonal components of the dielectric tensor. Default is (12, 12, 12).
            eps_offdiag (mp.Vector3): The off-diagonal components of the dielectric tensor. Default is (0, 0, 0).
            E_chi2_diag (mp.Vector3): The diagonal components of the second-order nonlinear susceptibility tensor. Default is (0, 0, 0).
            E_chi3_diag (mp.Vector3): The diagonal components of the third-order nonlinear susceptibility tensor. Default is (0, 0, 0).

        Returns:
            list: A list of mpb geometric objects defining the photonic crystal.
        """

        geometry = [
            #background
            mp.Block(
                size = mp.Vector3(mp.inf, mp.inf, height_supercell),
                material=mp.Medium(epsilon=eps_background)),
            #substrate
            mp.Block(
                size = mp.Vector3(mp.inf, mp.inf, height_supercell*0.5),
                center = mp.Vector3(0, 0, -height_supercell*0.25),
                material=mp.Medium(epsilon=eps_substrate)),
            #slab
            mp.Block(
                size = mp.Vector3(mp.inf, mp.inf, height_slab),
                material=mp.Medium(epsilon_diag=eps_diag,
                                   epsilon_offdiag = eps_offdiag,
                                   E_chi2_diag = E_chi2_diag,
                                   E_chi3_diag = E_chi3_diag)),
            #atom
            mp.Ellipsoid(size=mp.Vector3(e1,e2, mp.inf),
                         material=mp.Medium(epsilon=eps_atom),
                         center=mp.Vector3(0,0,0))
        ]
        return geometry
    
    

    def advanced_material_geometry(
        radius_1 = 0.2,
        epsilon_diag = mp.Vector3(12, 12, 12),
        epsilon_offdiag =  mp.Vector3(0, 0, 0),
        chi2_diag =  mp.Vector3(0,0,0),  
        chi3_diag =  mp.Vector3(0,0,0),
        eps_atom_1 = 1, 
        eps_background  = 1, 
        eps_substrate = 1,
        height_supercell = 4, 
        height_slab = 0.5
    )-> list:
        """
        Define an advanced material geometry for the photonic crystal.

        Args:
            radius_1 (float): The radius of the atom. Default is 0.2.
            epsilon_diag (mp.Vector3): The diagonal components of the dielectric tensor. Default is (12, 12, 12).
            epsilon_offdiag (mp.Vector3): The off-diagonal components of the dielectric tensor. Default is (0, 0, 0).
            chi2_diag (mp.Vector3): The diagonal components of the second-order nonlinear susceptibility tensor. Default is (0, 0, 0).
            chi3_diag (mp.Vector3): The diagonal components of the third-order nonlinear susceptibility tensor. Default is (0, 0, 0).
            eps_atom_1 (float): The dielectric constant of the atom. Default is 1.
            eps_background (float): The dielectric constant of the background. Default is 1.
            eps_substrate (float): The dielectric constant of the substrate. Default is 1.
            height_supercell (float): The height of the supercell. Default is 4.
            height_slab (float): The height of the slab. Default is 0.5.

        Returns:
            list: A list of geometric objects defining the photonic crystal.
        """

        
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
                material=mp.Medium(epsilon_diag=epsilon_diag, 
                                   epsilon_offdiag = epsilon_offdiag,
                                   E_chi2_diag = chi2_diag,
                                   E_chi3_diag = chi3_diag,)
                                ),
        )

        #atom 1
        geometry.append(mp.Cylinder(radius_1, 
                                        material=mp.Medium(epsilon=eps_atom_1),
                                        height=height_slab))
        return geometry

                            

    def plot_field(self, 
            target_polarization, 
            target_k_point, 
            target_frequency, 
            frequency_tolerance = 0.01, 
            k_point_max_distance = None,
            periods: int=1, 
            component: int = 2, 
            quantity: str = "real", 
            colorscale: str = 'RdBu',                  
            ):
        """
        Plot the field for a specific mode based on the given parameters.

        Args:
            target_polarization (str): The polarization of the target mode.
            target_k_point (tuple): The k-point of the target mode.
            target_frequency (float): The frequency of the target mode.
            frequency_tolerance (float): The tolerance for frequency similarity.
            periods (int): The number of periods to extract. Default is 1.
            component (int): The component of the field to plot (0 for x, 1 for y, 2 for z). Default is 2.
            quantity (str): The quantity to plot ('real', 'imag', or 'abs'). Default is 'real'.
            colorscale (str): The colorscale to use for the plot. Default is 'RdBu'.

        Returns:
            tuple: A tuple containing the electric field figure and the magnetic field figure.
        """
        target_modes = self.look_for_mode(target_polarization, target_k_point, target_frequency, freq_tolerance = frequency_tolerance, k_point_max_distance = k_point_max_distance)
        #print(len(target_modes))
        
        with suppress_output():
            self.run_dumb_simulation()
            md = mpb.MPBData(rectify=True, periods=periods, lattice=self.ms.get_lattice())
            eps = md.convert(self.ms.get_epsilon()) 

        z_points = eps.shape[2] // periods
        z_mid = eps.shape[2] // 2
        
        # Now take only epsilon in the center of the slab
        eps = eps[..., z_mid]
        # Calculate the midpoint between min and max of the permittivity (eps)
        min_eps, max_eps = np.min(eps), np.max(eps)
        midpoint = (min_eps + max_eps) / 2  # The level to be plotted

        fig_e = go.Figure()
        fig_h = go.Figure()
        dropdown_buttons_e = []
        dropdown_buttons_h = []
        
        num_plots = len(target_modes)
        
        for i, mode in enumerate(target_modes):
            # Initialize visibility status: False for all traces
            visible_status_e = [False] * (2 * num_plots)  # Each mode adds two traces
            visible_status_h = [False] * (2 * num_plots)

            visible_status_e[2 * i] = True  # Set the contour plot visible
            visible_status_h[2 * i] = True
            visible_status_e[2 * i + 1] = True  # Set the heatmap visible
            visible_status_h[2 * i + 1] = True  # Set the heatmap visible

            k_point = mode["k_point"]
            freq    = mode["freq"]
            polarization = mode["polarization"]

            # Take the specified component of the fields in the center of the slab
            
            e_field = mpb.MPBArray(mode["e_field"], lattice = self.ms.get_lattice(),  kpoint = mode["k_point"] )
            h_field = mpb.MPBArray(mode["h_field"], lattice = self.ms.get_lattice(),  kpoint = mode["k_point"])
            e_field = e_field[..., z_points // 2, component]
            h_field = h_field[..., z_points // 2, component]
            with suppress_output():
                e_field = md.convert(e_field) 
                h_field = md.convert(h_field)

            if quantity == "real":
                e_field = np.real(e_field)
                h_field = np.real(h_field)
            elif quantity == "imag":
                e_field = np.imag(e_field)
                h_field = np.imag(h_field)
            elif quantity == "abs":
                e_field = np.abs(e_field)
                h_field = np.abs(h_field)
            else:
                raise ValueError("Invalid quantity. Choose 'real', 'imag', or 'abs'.")

            # Add the contour plot for permittivity (eps) at the midpoint
            contour_e = go.Contour(z=eps.T,
                                contours=dict(
                                    start=midpoint,  # Start and end at the midpoint to ensure a single level
                                    end=midpoint,
                                    size=0.1,  # A small size to keep it as a single contour
                                    coloring='none'  # No filling
                                ),
                                line=dict(color='black', width=2),
                                showscale=False,
                                opacity=0.7,
                                visible=True if i ==  0  else False)  # Only the first mode is visible
            contour_h = contour_e  # Same contour for H-field figure

            # Add the contour trace
            fig_e.add_trace(contour_e)
            fig_h.add_trace(contour_h)

            # Add the heatmap for the electric field
            heatmap_e = go.Heatmap(z=e_field.T, colorscale=colorscale, zsmooth='best', opacity=0.9,
                                    showscale=True, visible= True if i == 0 else False)
            # Add the heatmap for the magnetic field
            heatmap_h = go.Heatmap(z=h_field.T, colorscale=colorscale, zsmooth='best', opacity=0.9,
                                    showscale=True, visible= True if i == 0 else False)

            # Add the heatmap trace
            fig_e.add_trace(heatmap_e)
            fig_h.add_trace(heatmap_h)

            

            data_str = f"Mode {i + 1} <br> k = [{k_point[0]:0.2f}, {k_point[1]:0.2f}], freq={freq:0.3f}, polarization={polarization}"
            if component == 0: 
                component_str = "x-component"
            elif component == 1:
                component_str = "y-component"
            else:
                component_str = "z-component"
            subtitle_e = f"E-field, {component_str}, {quantity}"
            subtitle_h = f"H-field, {component_str}, {quantity}"
            # Create a button for each field dataset for the dropdown
            dropdown_buttons_e.append(dict(label=f'Mode {i + 1}',
                                        method='update',
                                        args=[{'visible': visible_status_e},  # Update visibility for both eps and field
                                            {'title':f"{data_str}:<br> {subtitle_e}"}
                                        ]))
            dropdown_buttons_h.append(dict(label=f'Mode {i + 1}',
                                        method='update',
                                        args=[{'visible': visible_status_h},  # Update visibility for both eps and field
                                            {'title':f"{data_str}:<br> {subtitle_h}"}
                                        ]))

        # print(len(target_modes))
        k_point = target_modes[0]["k_point"]
        freq    = target_modes[0]["freq"]
        data_str = f"Mode {0} <br> k = [{k_point[0]:0.2f}, {k_point[1]:0.2f}], freq={freq:0.3f}, polarization={polarization}"
        
        fig_e.update_layout(
            updatemenus=[dict(active=0,
                            buttons=dropdown_buttons_e,
                            x=1.15, y=1.15,
                            xanchor='left', yanchor='top')],
            title=f"{data_str}:<br> {subtitle_e}",
            xaxis_showgrid=False, yaxis_showgrid=False,
            xaxis_zeroline=False, yaxis_zeroline=False,
            xaxis_visible=False, yaxis_visible=False,
            hovermode="closest",
            width=800, height=800,
            xaxis_title="X",
            yaxis_title="Y"
            
        )

        fig_h.update_layout(
            updatemenus=[dict(active=0,
                            buttons=dropdown_buttons_h,
                            x=1.15, y=1.15,
                            xanchor='left', yanchor='top')],
            title=f"{data_str}:<br> {subtitle_h}",
            xaxis_showgrid=False, yaxis_showgrid=False,
            xaxis_zeroline=False, yaxis_zeroline=False,
            xaxis_visible=False, yaxis_visible=False,
            hovermode="closest",
            width=800, height=800,
            xaxis_title="X",
            yaxis_title="Y"
        )

        return fig_e, fig_h
    
    

    def plot_field_components(self,
                            target_polarization,
                            target_k_point,
                            target_frequency,
                            frequency_tolerance=0.01,
                            k_point_max_distance=None,
                            periods: int = 1,
                            quantity: str = "real",
                            colorscale: str = 'RdBu',
                            )-> tuple:
        """
        Plot the field components (Ex, Ey, Ez) and (Hx, Hy, Hz) for specific modes with consistent color scales.

        Args:
            target_polarization (str): The polarization of the target mode.
            target_k_point (tuple): The k-point of the target mode.
            target_frequency (float): The frequency of the target mode.
            frequency_tolerance (float): The tolerance for frequency similarity.
            periods (int): The number of periods to extract. Default is 1.
            quantity (str): The quantity to plot ('real', 'imag', or 'abs'). Default is 'real'.
            colorscale (str): The colorscale to use for the plot. Default is 'RdBu'.

        Returns:
            tuple: A tuple containing the electric field figure and the magnetic field figure.
        """

        target_modes = self.look_for_mode(target_polarization, target_k_point, target_frequency,
                                        freq_tolerance=frequency_tolerance, k_point_max_distance=k_point_max_distance)
        print(f"Number of target modes found: {len(target_modes)}")

        with suppress_output():
            self.run_dumb_simulation()
            md = mpb.MPBData(rectify=True, periods=periods, lattice=self.ms.get_lattice())
            eps = md.convert(self.ms.get_epsilon())

        z_points = eps.shape[2] // periods
        z_mid = eps.shape[2] // 2

        # Now take only epsilon in the center of the slab
        eps = eps[..., z_mid]
        # Calculate the midpoint between min and max of the permittivity (eps)
        min_eps, max_eps = np.min(eps), np.max(eps)
        midpoint = (min_eps + max_eps) / 2  # The level to be plotted

        fig_e = make_subplots(rows=1, cols=3, subplot_titles=("Ex", "Ey", "Ez"))
        fig_h = make_subplots(rows=1, cols=3, subplot_titles=("Hx", "Hy", "Hz"))
        dropdown_buttons_e = []
        dropdown_buttons_h = []

        # For each mode, calculate separate min and max values for Ex, Ey, Ez (and similarly Hx, Hy, Hz)
        for i, mode in enumerate(target_modes):
            # Get field arrays for this mode
            e_field_array = mpb.MPBArray(mode["e_field"], lattice=self.ms.get_lattice(), kpoint=mode["k_point"])
            h_field_array = mpb.MPBArray(mode["h_field"], lattice=self.ms.get_lattice(), kpoint=mode["k_point"])

            # Extract field components in the center of the slab
            e_field_x = e_field_array[..., z_points // 2, 0]  # Shape (Nx, Ny)
            e_field_y = e_field_array[..., z_points // 2, 1]  # Shape (Nx, Ny)
            e_field_z = e_field_array[..., z_points // 2, 2]  # Shape (Nx, Ny)
            h_field_x = h_field_array[..., z_points // 2, 0]  # Shape (Nx, Ny)
            h_field_y = h_field_array[..., z_points // 2, 1]  # Shape (Nx, Ny)
            h_field_z = h_field_array[..., z_points // 2, 2]  # Shape (Nx, Ny)

            with suppress_output():
                e_field_x = md.convert(e_field_x)
                e_field_y = md.convert(e_field_y)
                e_field_z = md.convert(e_field_z)
                h_field_x = md.convert(h_field_x)
                h_field_y = md.convert(h_field_y)
                h_field_z = md.convert(h_field_z)

            e_field = np.stack([e_field_x, e_field_y, e_field_z], axis=-1)
            h_field = np.stack([h_field_x, h_field_y, h_field_z], axis=-1)                                    

            # Select quantity to display (real, imag, abs)
            if quantity == "real":
                e_field = np.real(e_field)
                h_field = np.real(h_field)
            elif quantity == "imag":
                e_field = np.imag(e_field)
                h_field = np.imag(h_field)
            elif quantity == "abs":
                e_field = np.abs(e_field)
                h_field = np.abs(h_field)
            else:
                raise ValueError("Invalid quantity. Choose 'real', 'imag', or 'abs'.")

            # Calculate the component-specific min/max for E and H fields of this mode
            e_min = np.min(e_field)
            e_max = np.max(e_field)
            h_min = np.min(h_field)
            h_max = np.max(h_field)

            # Components of the E and H fields
            Ex, Ey, Ez = e_field[..., 0], e_field[..., 1], e_field[..., 2]
            Hx, Hy, Hz = h_field[..., 0], h_field[..., 1], h_field[..., 2]

            # Define visibility settings per mode, including contours as always visible
            visible_status_e = [False] * (len(target_modes) * 6)  # 3 components per mode, with contour for each
            visible_status_h = [False] * (len(target_modes) * 6)
            # Make the contour visible by default

            


            # Make this mode's components and the corresponding contour visible in the initial layout 
            for j in range(6):
                visible_status_e[6*i + j] = True
                visible_status_h[6*i + j] = True


            # Add contour traces for permittivity to each subplot of fig_e and fig_h for this mode
            fig_e.add_trace(go.Contour(z=eps.T, contours=dict(start=midpoint, end=midpoint, size=0.1, coloring='none'), line=dict(color='black', width=2), showscale=False, opacity=0.7, showlegend=False, visible=True if i == len(target_modes)-1 else False), row=1, col=1)
            fig_e.add_trace(go.Contour(z=eps.T, contours=dict(start=midpoint, end=midpoint, size=0.1, coloring='none'), line=dict(color='black', width=2), showscale=False, opacity=0.7, showlegend=False, visible=True if i == len(target_modes)-1 else False), row=1, col=2)
            fig_e.add_trace(go.Contour(z=eps.T, contours=dict(start=midpoint, end=midpoint, size=0.1, coloring='none'), line=dict(color='black', width=2), showscale=False, opacity=0.7, showlegend=False, visible=True if i == len(target_modes)-1 else False), row=1, col=3)

            fig_h.add_trace(go.Contour(z=eps.T, contours=dict(start=midpoint, end=midpoint, size=0.1, coloring='none'), line=dict(color='black', width=2), showscale=False, opacity=0.7, showlegend=False, visible=True if i == len(target_modes)-1 else False), row=1, col=1)
            fig_h.add_trace(go.Contour(z=eps.T, contours=dict(start=midpoint, end=midpoint, size=0.1, coloring='none'), line=dict(color='black', width=2), showscale=False, opacity=0.7, showlegend=False, visible=True if i == len(target_modes)-1 else False), row=1, col=2)
            fig_h.add_trace(go.Contour(z=eps.T, contours=dict(start=midpoint, end=midpoint, size=0.1, coloring='none'), line=dict(color='black', width=2), showscale=False, opacity=0.7, showlegend=False, visible=True if i == len(target_modes)-1 else False), row=1, col=3)

            # Add Ex, Ey, Ez with shared colorbar limits for the E field of this mode
            fig_e.add_trace(go.Heatmap(z=Ex.T, colorscale=colorscale, showscale=False, zmin=e_min, zmax=e_max, visible=True if i == len(target_modes)-1 else False, zsmooth="best", opacity=0.8), row=1, col=1)
            fig_e.add_trace(go.Heatmap(z=Ey.T, colorscale=colorscale, showscale=False, zmin=e_min, zmax=e_max, visible=True if i == len(target_modes)-1 else False, zsmooth="best", opacity=0.8), row=1, col=2)
            fig_e.add_trace(go.Heatmap(z=Ez.T, colorscale=colorscale, showscale=True, colorbar=dict(title="E-field", len=0.75, thickness=15), zmin=e_min, zmax=e_max, visible=True if i == len(target_modes)-1 else False, zsmooth="best", opacity=0.8), row=1, col=3)

            # Add Hx, Hy, Hz with shared colorbar limits for the H field of this mode
            fig_h.add_trace(go.Heatmap(z=Hx.T, colorscale=colorscale, showscale=False, zmin=h_min, zmax=h_max, visible=True if i == len(target_modes)-1 else False, zsmooth="best", opacity=0.8), row=1, col=1)
            fig_h.add_trace(go.Heatmap(z=Hy.T, colorscale=colorscale, showscale=False, zmin=h_min, zmax=h_max, visible=True if i == len(target_modes)-1 else False, zsmooth="best", opacity=0.8), row=1, col=2)
            fig_h.add_trace(go.Heatmap(z=Hz.T, colorscale=colorscale, showscale=True, colorbar=dict(title="H-field", len=0.75, thickness=15), zmin=h_min, zmax=h_max, visible=True if i == len(target_modes)-1 else False, zsmooth="best", opacity=0.8), row=1, col=3)
            
            

            # Dropdown data for E-field
            k_point = mode["k_point"]
            freq = mode["freq"]
            polarization = mode["polarization"]
            mode_description = f"Mode {i + 1}<br>k = [{k_point[0]:0.2f}, {k_point[1]:0.2f}], freq={freq:0.3f}, polarization={polarization}"
            
            dropdown_buttons_e.append(
                dict(label=f"Mode {i + 1}",
                    method='update',
                    args=[{'visible': visible_status_e},
                        {'title': f"{mode_description}: {quantity} of E-field components"}]))

            dropdown_buttons_h.append(
                dict(label=f"Mode {i + 1}",
                    method='update',
                    args=[{'visible': visible_status_h},
                        {'title': f"{mode_description}: {quantity} of H-field components"}]))

        # Layout and color settings
        fig_e.update_layout(
            title=f"{mode_description}: {quantity} of E-field components",
            updatemenus=[dict(
                active=len(target_modes) - 1,
                buttons=dropdown_buttons_e)],
            coloraxis=dict(colorbar=dict(len=0.75)),
            width=1200, height=400,
            xaxis_showgrid=False, yaxis_showgrid=False,
            xaxis_zeroline=False, yaxis_zeroline=False,
            hovermode="closest"
        )

        fig_h.update_layout(
            title=f"{mode_description}: {quantity} of H-field components",
            updatemenus=[dict(
                active=len(target_modes) - 1,
                buttons=dropdown_buttons_h)],
            coloraxis=dict(colorbar=dict(len=0.75)),
            width=1200, height=400,
            xaxis_showgrid=False, yaxis_showgrid=False,
            xaxis_zeroline=False, yaxis_zeroline=False,
            hovermode="closest"
        )

        # Final adjustments
        fig_e.update_xaxes(showticklabels=False)
        fig_e.update_yaxes(showticklabels=False)
        fig_h.update_xaxes(showticklabels=False)
        fig_h.update_yaxes(showticklabels=False)

        fig_e.update_layout(yaxis_scaleanchor="x", yaxis_scaleratio=1)
        fig_h.update_layout(yaxis_scaleanchor="x", yaxis_scaleratio=1)
        fig_e.update_layout(yaxis2_scaleanchor="x2", yaxis2_scaleratio=1)
        fig_h.update_layout(yaxis2_scaleanchor="x2", yaxis2_scaleratio=1)
        fig_e.update_layout(yaxis3_scaleanchor="x3", yaxis3_scaleratio=1)
        fig_h.update_layout(yaxis3_scaleanchor="x3", yaxis3_scaleratio=1)


        fig_e.update_xaxes(title_text="X-axis", row=1, col=1)
        fig_e.update_yaxes(title_text="Y-axis", row=1, col=1)
        fig_e.update_xaxes(title_text="X-axis", row=1, col=2)
        fig_e.update_yaxes(title_text="Y-axis", row=1, col=2)
        fig_e.update_xaxes(title_text="X-axis", row=1, col=3)
        fig_e.update_yaxes(title_text="Y-axis", row=1, col=3)

        fig_h.update_xaxes(title_text="X-axis", row=1, col=1)
        fig_h.update_yaxes(title_text="Y-axis", row=1, col=1)
        fig_h.update_xaxes(title_text="X-axis", row=1, col=2)
        fig_h.update_yaxes(title_text="Y-axis", row=1, col=2)
        fig_h.update_xaxes(title_text="X-axis", row=1, col=3)
        fig_h.update_yaxes(title_text="Y-axis", row=1, col=3)

        return fig_e, fig_h


    



# %%
