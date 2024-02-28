from astroquery.mast import Observations

# print(Observations.list_missions()[2])
# jwst_data = Observations.query_mast(mission_name='JWST')

# obs_table = Observations.query_criteria(mission="JWST", target_name="M87")
# jwst_data = obs_table.get_product_list(product_type="SCIENCE")  # Example for JWST

meta_table = Observations.get_metadata("observations")


import lightkurve as lk
pixelfile = lk.search_targetpixelfile("Trappist-1")[1].download()
lc = pixelfile.to_lightcurve(method="pld").remove_outliers().flatten()
period = lc.to_periodogram("bls").period_at_max_power
lc.fold(period).truncate(-.5, -.25).scatter()
# pixelfile.animate()
