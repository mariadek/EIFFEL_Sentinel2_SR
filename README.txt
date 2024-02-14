The development of the EIFFEL_Sentinel2_SR code is part of the EIFFEL Horizon Project. This project has received funding from the 
European Union’s Horizon 2020 research and innovation programme under Grant Agreement No 101003518.

https://cordis.europa.eu/project/id/101003518

# Create an anaconda environment from the EIFFEL_S2_SR_env.yml file:

  conda env create -f EIFFEL_S2_SR_env.yml

# Activate the new environment:

  conda activate EIFFEL_S2_SR

# Execute python script

  python EIFFEL_Sen2_SR_Predict.py --input <Sentinel2 zip filepath>

  Example: python EIFFEL_Sen2_SR_Predict.py --input E:\EIFFEL_S2_SR\S2A_MSIL2A_20210910T101031_N0301_R022_T35WMP_20210910T114151.zip
  
  The output is a 12-band Geotiff file stored in the same directory as the input file and named as SR_<input_filename>.tiff 
  (i.e. SR_S2A_MSIL2A_20210910T101031_N0301_R022_T35WMP_20210910T114151.tiff) 
  with the following configuration (band descriptions are included in the output file):
  	01: B4 (665 nm)
	02: B3 (560 nm)
	03: B2 (490 nm)
	04: B8 (842 nm)
	05: B5 (705 nm)
	06: B6 (740 nm)
	07: B7 (783 nm)
	08: B8A (865 nm)
	09: B11 (1610 nm)
	10: B12 (2190 nm)
	11: B1 (443 nm)
	12: B9 (945 nm)
	
