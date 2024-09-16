import os
from photonic_crystal import PhotonicCrystal
import plotly.graph_objects as go

def main():
    # Create the output directory for HTML files if it doesn't exist
    if not os.path.exists('pics'):
        os.makedirs('pics')

    # Initialize two photonic crystals with different lattice types
    pc_square = PhotonicCrystal(lattice_type='square', num_bands=8, pickle_id='square_lattice')
    pc_triangular = PhotonicCrystal(lattice_type='triangular', num_bands=8, pickle_id='triangular_lattice')

    # Run the simulations for both crystals
    pc_square.run_simulation(type='both')
    pc_square.extract_data(periods=5)
    
    pc_triangular.run_simulation(type='both')
    pc_triangular.extract_data(periods=5)

    # Generate interactive figures for both crystals
    fig_square_eps = go.Figure(layout=go.Layout(width=500, height=500))
    fig_triangular_eps = go.Figure(layout=go.Layout(width=500, height=500))
    fig_square_bands = go.Figure(layout=go.Layout(width=500, height=500))
    fig_triangular_bands = go.Figure(layout=go.Layout(width=500, height=500))
    fig_square_field = go.Figure(layout=go.Layout(width=500, height=500))
    fig_triangular_field = go.Figure(layout=go.Layout(width=500, height=500))

    # Plot epsilon (dielectric) distribution for both crystals
    pc_square.plot_epsilon_interactive(fig=fig_square_eps, title='Square Lattice Epsilon')
    pc_triangular.plot_epsilon_interactive(fig=fig_triangular_eps, title='Triangular Lattice Epsilon')

    # Plot band structures for both crystals (TE and TM polarizations)
    pc_square.plot_bands_interactive(polarization='te', fig=fig_square_bands, title='Square Lattice Bands (TE)', color='blue')
    pc_square.plot_bands_interactive(polarization='tm', fig=fig_square_bands, title='Square Lattice Bands (TM)', color='red')
    pc_triangular.plot_bands_interactive(polarization='te', fig=fig_triangular_bands, title='Triangular Lattice Bands (TE)', color='blue')
    pc_triangular.plot_bands_interactive(polarization='tm', fig=fig_triangular_bands, title='Triangular Lattice Bands (TM)', color='red')

    # Get the high symmetry k-points for field plotting
    k_points_square = pc_square.get_high_symmetry_points()
    k_points_triangular = pc_triangular.get_high_symmetry_points()

    # Plot the electric field at a specific k-point (M point) for both crystals
    pc_square.plot_field_interactive(k_point=k_points_square['M'], fig=fig_square_field, title='Square Lattice Field at M Point')
    pc_triangular.plot_field_interactive(k_point=k_points_triangular['M'], fig=fig_triangular_field, title='Triangular Lattice Field at M Point')

    # Save the plots to the 'pics' directory as HTML files
    fig_square_eps.write_html('pics/square_lattice_epsilon.html')
    fig_triangular_eps.write_html('pics/triangular_lattice_epsilon.html')
    fig_square_bands.write_html('pics/square_lattice_bands.html')
    fig_triangular_bands.write_html('pics/triangular_lattice_bands.html')
    fig_square_field.write_html('pics/square_lattice_field.html')
    fig_triangular_field.write_html('pics/triangular_lattice_field.html')

    print("Simulation complete. HTML files saved in 'pics' directory.")

if __name__ == '__main__':
    main()
