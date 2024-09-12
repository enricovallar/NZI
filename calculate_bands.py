import sys
import PhotonicCrystal
import plotly.graph_objects as go
import json

def calculate_bands(lattice_geometry="square", cylinder_radius=0.2, sim_type="both"):
    if lattice_geometry == "square":
        lattice = PhotonicCrystal.square_lattice()
    elif lattice_geometry == "triangular":
        lattice = PhotonicCrystal.triangular_lattice()
    else:
        raise ValueError("Unsupported lattice geometry")

    geometry = PhotonicCrystal.basic_geometry(radius=cylinder_radius, eps=12.0)

    pc = PhotonicCrystal(geometry=geometry, geometry_lattice=lattice)
    pc.run_simulation(type=sim_type)
    pc.extract_data(periods=5)
    
    fig_eps = go.Figure(layout=go.Layout(width=500, height=500))
    fig_bands = go.Figure(layout=go.Layout(width=500, height=500))

    pc.plot_epsilon_interactive(title="Lattice", fig=fig_eps)
    pc.plot_bands_interactive(polarization='te', title='Square Lattice TE and TM Bands', color='blue', fig=fig_bands)
    pc.plot_bands_interactive(polarization='tm', title='Square Lattice TE and TM Bands', color='red', fig=fig_bands)

    return fig_eps.to_html(full_html=False), fig_bands.to_html(full_html=False)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 calculate_bands.py <lattice_geometry> <cylinder_radius>")
        sys.exit(1)

    lattice_geometry = sys.argv[1]
    cylinder_radius = sys.argv[2]

    try:
        fig_eps, fig_bands = calculate_bands(lattice_geometry, float(cylinder_radius))
        result = {
            "fig_eps": fig_eps,
            "fig_bands": fig_bands
        }
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)