import os
import sys

def probe_output_to_vtk(probe_pre_path: str, probe_out_path: str, probe_out_xdmf: str):
    '''
    TS4310 probe output file probe_output.hdf5 cannot be read directly into paraview
    as there is no longer ts probe reader, so cannot use previous probe reading methods.
    
    So we need to create XDMF for the probe output that contains all the measured parameters
    at different timesteps at the specified nodes/boundaries.

    Parameters:
    -----------
    probe_pre_path : str
        Path of the pre-processed probe HDF5 file before the simulation is run. E.g. probe_wing.hdf5 that gets
        generated BEFORE the simulation is run. Contains data: ['aname_list', 'area_avg_aname_list', 'ax', 'ay', 'az', 'cells', 'mass_avg_aname_list', 'node_ids', 'pts', 'weights']

    probe_out_path : str
        Path to the output probe data that contains measurements of all quantities measured during the simulation
        at all nodes and time steps specified in the probes.py file. Contains data e.g. pstat, rovz, x, y, z, ro etc. 

    probe_out_xdmf : str
        Path of output probe data converted into XDMF file which can be visualised in paraview.
    '''

    #==============
    # input files
    #==============

    # HDF5 file created when probe.py was pre-processed BEFORE running the simulation
    cmd = f'python /usr/local/software/turbostream/ts4310/a100_pv513/turbostream_4/script/ts/util/make_probe_xdmf.py {probe_pre_path} {probe_out_path} {probe_out_xdmf}'
    
    print(f"Running command: {cmd}")
    result = os.system(cmd)
    
    if result == 0:
        print(f"Successfully created XDMF file: {probe_out_xdmf}")
    else:
        print(f"Error creating XDMF file. Command returned exit code: {result}")

def main():
    """Main function that uses current working directory as parent path"""
    # Get current working directory
    parent_path = os.getcwd()
    print(f"Using current working directory as parent path: {parent_path}")
    
    # Check if required files exist
    probe_pre_file = os.path.join(parent_path, "probes.hdf5")
    probe_out_file = os.path.join(parent_path, "probe_out.hdf5")
    
    if not os.path.exists(probe_pre_file):
        print(f"Warning: probes.hdf5 not found in current directory: {probe_pre_file}")
        print("Please ensure you're in the correct directory with probe files.")
        sys.exit(1)
    
    if not os.path.exists(probe_out_file):
        print(f"Warning: probe_out.hdf5 not found in current directory: {probe_out_file}")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Set output XDMF file path
    probe_out_xdmf_file = os.path.join(parent_path, "probe_out.xdmf")
    
    # Call the conversion function
    probe_output_to_vtk(
        probe_pre_path=probe_pre_file,
        probe_out_path=probe_out_file,
        probe_out_xdmf=probe_out_xdmf_file
    )

if __name__ == '__main__':
    main()