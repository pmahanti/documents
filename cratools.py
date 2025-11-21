from re import I
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio
import matplotlib.pyplot as plt

import rasterio.plot
import rasterio.mask
import scipy.integrate
import scipy.interpolate
from scipy.ndimage import gaussian_filter1d
import shapely.geometry

import uncertainties as unp
from uncertainties import unumpy 

import scipy.optimize
import scipy.signal
import copy

def fit_crater_rim(geom, 
                   dem_src,
                   crs,
                   orthophoto,
                   inner_radius = 0.8,
                   outer_radius = 1.2,
                   remove_ext = True, 
                   plot = False):
    
    orthophoto = rio.open(orthophoto, "r")
    #Compute radius and center of original crater


    try:
        radius = np.sqrt(geom.area / np.pi)
    except Exception as e:
        return np.nan, (np.nan, np.nan, np.nan)
    center = geom.centroid
    center = np.array([center.x, center.y]) #In map coordinates (x,y)

    if remove_ext:
        (out_image, out_transform), (ortho_image, ortho_transform) = remove_external_topography(geom = geom,
                                                                dem_src=dem_src, orthophoto=orthophoto,
                                                                crs=crs)
    else:

        #Create bounding box with width of 3 * R
        buffered = shapely.geometry.box(center[0] - radius * 3, center[1] - radius * 3, center[0] + radius * 3, center[1] + radius * 3)

        mask = shapely.geometry.MultiPolygon([buffered])

        (out_image, out_transform) = rio.mask.mask(dem_src, mask, crop = True, all_touched=True)


    # Define azimuthal range and radius range for determining crater rim.
        
    azimuths = np.arange(0.,359.,5)
    r_dist = np.arange(inner_radius, outer_radius, .01)
    
    # Compute matrix with rows of distance (units of R) and cols of az (0-359 deg)
    E_rim = compute_E_matrix(image=out_image,
                     transform = out_transform,
                     center = center,
                     azimuths=azimuths,
                     r_dist=r_dist)




    circlefit_idx = np.where(np.isclose(r_dist, 1.0, atol=1e-8))[0]
    rim_idx = np.full(E_rim.shape[1], circlefit_idx)   # same length as number of columns
    count = 0
    for idx, az in enumerate(azimuths):
        #profile is all radii at this azimuth iterator

        prof = E_rim[:,idx]
        (local_max_idx) = scipy.signal.find_peaks(prof, prominence=1e-5 * radius)
        if local_max_idx[0].size == 0:
            # if count % 3 == 0:
            #     continue
            
        # #     #If no obvious rim, instead use location where slope changes the least
            second_deriv = np.gradient(np.gradient(prof))
            rim_idx[idx] = np.argmin(second_deriv)

            # testing slope minimum instead of f''
            # first_deriv = np.gradient(prof)
            # rim_idx[idx] = np.argmin(abs(first_deriv))

        # #     drop that profile altogether
        #     # rim_idx[idx] = int(-1)
            
        else:
            rim_idx[idx] = local_max_idx[0][np.argmax(local_max_idx[1]["prominences"])]

        count += 1


    # Convert polar to cartesian
    valid_mask = rim_idx >= 0
    rim_r_pos = np.full_like(rim_idx, np.nan, dtype=float)
    rim_r_pos[valid_mask] = r_dist[rim_idx[valid_mask]]

    #rim_r_pos = r_dist[rim_idx]
    rim_r_pos[rim_r_pos < inner_radius] = np.nan
    rim_r_pos[rim_r_pos > outer_radius] = np.nan

    rim_x = rim_r_pos * np.cos(azimuths * np.pi/180)
    rim_y = rim_r_pos * np.sin(azimuths * np.pi/180)

    # Find the planform circle of best fit for each of these rim heights
    def obj_circle(params, x, y):
        x0, y0, rad = params

        return rad**2 - ((x - x0)**2 + (y - y0)**2)

    fit_result = scipy.optimize.least_squares(obj_circle, x0 = [0,0,.5], args = (rim_x[~np.isnan(rim_x)], rim_y[~np.isnan(rim_x)]), gtol= 1e-10)
    (x0_fit, y0_fit, rad_fit_fac) = fit_result.x

    # Compute fit error (approx Hessian as J'J)
    J = fit_result.jac
    cov = np.linalg.inv(J.T @ J)
    err = np.sqrt(np.diag(cov))

    # Update original parameters using fit
    radius_fit = radius * rad_fit_fac
    center_fit = [center[0] + x0_fit * radius, center[1] +  y0_fit * radius] #Geopandas/Shapely does x,y by default


    new_crater = shapely.geometry.Point(center_fit[0], center_fit[1]).buffer(radius_fit)

    if plot:
        print(f"{center[1]}, {center[0]}")
        fig, axs = plt.subplots(2,2, figsize = (20,20))
        axs[0,0].imshow(out_image[0,:,:])

        x_coord = rim_x * radius + center[0]
        y_coord = rim_y * radius + center[1]

        x_pix_coord = []
        y_pix_coord = []

        for i in range(len(x_coord)):
            (xformed_x, xformed_y) = ~out_transform * (x_coord[i], y_coord[i])
            x_pix_coord.append(xformed_x)
            y_pix_coord.append(xformed_y)

        axs[0,0].scatter(x_pix_coord, y_pix_coord, marker='.', c='red')
        axs[0,0].set_title("DEM")

        E_mat = compute_E_matrix(image=out_image,
                     transform = out_transform, center = center_fit)
        axs[0,1].imshow(E_mat, aspect='auto')
        axs[0,1].plot(azimuths[~np.isnan(rim_x)]/5, rim_idx[~np.isnan(rim_y)]+75, c='red')

        axs[0,1].set_title("radial profile matrix")
        axs[0,1].axhline(rad_fit_fac*100)
        axs[0,1].set_xlabel("azimuth Angle")
        axs[0,1].set_ylabel("radial index")

        stack_profile = np.zeros_like(E_mat[:,0])
        count = 0

        for i in range(0,71):

            prof = E_mat[:,i]
            stack_profile += E_mat[:,i]
            rad_norm = np.arange(0.01, 3.01, .01)
            if i % 1 == 0:
                #plot profiles vertically offset
                axs[1,0].plot(rad_norm,prof*5+count*1.3, color = "black", linewidth = 1)
                rim_range = prof[(int(np.where(np.isclose(rad_norm, inner_radius, atol=1e-8))[0])):int(np.where(np.isclose(rad_norm, outer_radius, atol=1e-8))[0])]
                (peak) = scipy.signal.find_peaks(rim_range, prominence= 1e-5 * radius)
                if peak[0].size == 0:
                    # if count % 3 == 0:
                    # #     continue
                    second_deriv = np.gradient(np.gradient(rim_range))
                    peak = int(np.argmin(second_deriv) + inner_radius/.01)
                    # first_deriv = np.gradient(rim_range)
                    # peak = int(np.argmin(np.abs(first_deriv)) + inner_radius/.01)

                    c='red'
                    label="f''"
                else:
                    peak = int(peak[0][np.argmax(peak[1]["prominences"])] + inner_radius/.01)
                    c='blue'
                    label="max"
                #color^^^ indicates which method found the rim position 
                axs[1,0].plot(rad_norm[peak], prof[peak]*5+count*1.3, marker=7, c=c)
            count += 1   
        # avg_profile = stack_profile/count
        # avg_rim_range = avg_profile[(int(np.where(np.isclose(rad_norm, inner_radius, atol=1e-8))[0])):int(np.where(np.isclose(rad_norm, outer_radius, atol=1e-8))[0])]
        # (avg_peak)= scipy.signal.find_peaks(avg_rim_range, prominence=1e-5 * radius)
        # #displaying where the peak was found for comparison with circle diam
        # if avg_peak[0].size == 0:
        #     second_deriv = np.gradient(np.gradient(avg_rim_range))
        #     avg_peak = int(np.argmin(second_deriv) + inner_radius/.01)
        # else:
        #     avg_peak = int(avg_peak[0][np.argmax(avg_peak[1]["prominences"])] + inner_radius/.01)

        # axs[1,0].plot(rad_norm, avg_profile, color = "blue", linestyle = "-.")
        # axs[1,0].plot(rad_norm[avg_peak], avg_profile[avg_peak], 'bo', label='rim' )
        # axs[1,0].plot(second_deriv)
        axs[1,0].set_xlabel("Normalized Radius")
        axs[1,0].set_ylabel("Elevation")
        axs[1,0].axvline(rad_fit_fac, label="radius of best-fit circle")
        axs[1,0].legend()

        craters = [geom, new_crater]
        df_crater_new = gpd.GeoDataFrame(index=[0,1], crs=crs, geometry=craters)
        print(df_crater_new)  
        df_crater_new["version"] = ["Original Rim", "Updated Rim"]

        df_crater_new.plot(ax = axs[1,1], column = "version", facecolor = "none", legend = True, linestyle='--')
        axs[1,1].set_xlim(center[0] - radius * 3, center[0] + radius * 3,)
        axs[1,1].set_ylim(center[1] - radius * 3, center[1] + radius * 3,)

        extent=rasterio.plot.plotting_extent(out_image, out_transform)

        if extent[0] > extent[2]:
            extent = (extent[2], extent[1], extent[0], extent[3])

        extent = [center[0] - radius * 3, center[0] + radius * 3, center[1] - radius * 3, center[1] + radius * 3]

        #axs[1,1].imshow(out_image[0,:,:], extent=extent,  cmap="pink")
        rasterio.plot.show(ortho_image, extent=extent, cmap='gray', ax=axs[1,1])
        axs[1,1].scatter(x_coord, y_coord, color = 'w', marker='.', alpha=0.5)
        plt.title(f"Center: {center}; Diameter: {int(radius * 2)}")
        plt.savefig(f"/Users/asonke/Library/CloudStorage/OneDrive-ArizonaStateUniversity/SLC/figs/{str(center)}.png")
        plt.show()
    
    return new_crater, err * radius

def remove_external_topography(geom, dem_src, orthophoto, crs):

    #Compute radius and center of original crater
    radius = np.sqrt(geom.area / np.pi)
    center = geom.centroid
    center = np.array([center.x, center.y]) #In map coordinates

    #Create bounding box with width of 3 * R
    buffered = shapely.geometry.box(center[0] - radius * 3.2, center[1] - radius * 3.2, center[0] + radius * 3.2, center[1] + radius * 3.2)

    mask = shapely.geometry.MultiPolygon([buffered])

    (out_image, out_transform) = rio.mask.mask(dem_src, mask.geoms, crop = True, all_touched=True)
    (ortho_image, ortho_transform) = rio.mask.mask(orthophoto, mask.geoms, crop = True, all_touched=True)


    (num_rows, num_cols) = out_image[0].shape

    # Define azimuthal range and radius range for determining crater rim.
        
    azimuths = np.arange(0.,359.,5)
    r_dist = np.array([3])
    
    # Compute matrix with rows of distance (units of R) and cols of az (0-359 deg)
    E_rim = compute_E_matrix(image=out_image,
                    transform = out_transform,
                    center = center,
                    azimuths=azimuths,
                    r_dist=r_dist)[0]
    
    xs = 3 * radius *  np.cos(azimuths * np.pi/180) + center[0]
    ys = 3 * radius * np.sin(azimuths * np.pi/180) + center[1]

    #print(xs)
    #print(ys)

    (ys_idx,xs_idx) = ~out_transform * (xs,ys)

    def plane_obj(params, x=xs_idx, y=ys_idx, elev=E_rim):
        [a,b,c] = params
        plane = a + b*x + c*y
        return elev - plane
    
    try:
        fit_result = scipy.optimize.least_squares(plane_obj, x0 = [1,1,1], gtol= 1e-10)
        [a, b, c] = fit_result.x
    except ValueError as e:
        print(e)
        [a, b, c] = [np.nanmean(E_rim), 0.0, 0.0]

    #Perform correction

    yy,xx = np.meshgrid(range(num_cols), range(num_rows))

    out_image[0] -= a + b * xx + c * yy

    return (out_image, out_transform), (ortho_image, ortho_transform)


def compute_E_matrix(image,
                     transform,
                     center,
                     r_dist = np.arange(0.01, 3.01, .01),
                     azimuths = np.arange(0.,359.,5)):    
    

    #Convert metric coordinates of center into row, col
    (row_center, col_center) = ~transform * center 
    (_, row, col) = image.shape

    row_s = np.linspace(-row_center / row * 6,(row -row_center) / row * 6, row)
    col_s = np.linspace(-col_center / col * 6,(col -col_center) / col * 6, col)

    interpolator_spline = scipy.interpolate.RectBivariateSpline(row_s,col_s,image[0,:, :], kx = 5, ky = 5)

    azaz, rr = np.meshgrid(azimuths, r_dist)
    azaz *= np.pi/180

    xs = rr * np.cos(azaz)
    ys = rr * np.sin(azaz)

    E_mat = interpolator_spline.ev(-ys, xs)

    # TESTING 1-D smoothing spline along each profile
    # r_dist is the x-axis for the spline
    r = np.asarray(r_dist)
    E_sm = np.array(E_mat, copy=True)


    k = 3   # degree, 1 <= k <= 5
    s = 2.0 # smoothing parameter

    for j in range(E_mat.shape[1]):  # for each azimuth (column)
        y = E_mat[:, j]
        try:
            spline1d = scipy.interpolate.UnivariateSpline(r, y, s=s, k=k)
            E_sm[:, j] = spline1d(r)  # evaluate at full r (fills NaNs too)
        except Exception:
            # if spline fitting fails for some reason, keep original column
            E_sm[:, j] = y
    else:
        # not enough points to fit; leave column unchanged
        E_sm[:, j] = y

    return E_sm

def compute_depth_diameter_ratio(geom, filename_dem, crs, orthophoto, diam_err = 0.1, remove_ext = True, plot = True):
    orthophoto = rio.open(orthophoto, "r")
    #orthophoto.nodata = np.nan
    with rio.open(filename_dem, 'r') as src:
        # src.nodata = np.nan
        if remove_ext:
            (out_image, out_transform), (ortho_image, ortho_transform) = remove_external_topography(geom = geom,
                                        dem_src=src,
                                        orthophoto=orthophoto,
                                        crs=crs)
        
            out_meta = copy.deepcopy(src.meta)
            out_meta.update({"driver": "GTiff",
                        "height": out_image.shape[1],
                        "width": out_image.shape[2],
                        "transform": out_transform})

            with rio.open("temp.IMG", "w", **out_meta) as dest: #Write to temporary file so we have proper georefrencing.
                dest.write(out_image)


            with rio.open("temp.IMG", "r" ) as src_mod:
                try:
                    mask = shapely.geometry.MultiPolygon([geom])
                except Exception as e:
                    mask = geom

                (out_image, out_transform) = rio.mask.mask(src_mod, mask.geoms, crop = True, all_touched=True, nodata = np.nan)
                out_image = out_image[0,:,:]
        
        else:
            try:
                mask = shapely.geometry.MultiPolygon([geom])
            except Exception as e:
                mask = geom

            (out_image, out_transform) = rio.mask.mask(src, mask.geoms, crop = True, all_touched=True, nodata = np.nan)
            out_image = out_image[0,:,:]

        if plot:
            plt.imshow(out_image)
            plt.colorbar()
            plt.show()

        image_perimeter = copy.deepcopy(out_image)
        test_image = np.pad(out_image, 1, constant_values = np.nan)

        for i in range(1, test_image.shape[0]-1):
            for j in range(1, test_image.shape[1]-1):
                subimage = test_image[i-1:i+2, j-1:j+2]

                if not np.any(np.isnan(subimage)):
                    #print("first case:")
                    #print(subimage)
                    image_perimeter[i-1,j-1] = np.nan
                
        if plot:
            plt.figure()
            plt.imshow(image_perimeter)
            plt.show()

        perim_vals = image_perimeter.flatten()
        perim_vals = perim_vals[np.invert(np.isnan(perim_vals))]

        rim_height = np.mean(perim_vals)
        rim_height_uncertainty = np.std(perim_vals)
        rim_height = unp.ufloat(rim_height, rim_height_uncertainty)
        floor_height = np.nanmin(out_image)

        depth = rim_height - floor_height

        diam = np.sqrt(geom.area/np.pi) * 2
        diam = unp.ufloat(diam, 0)

        ratio = depth/diam

    return ratio, depth, diam, rim_height, floor_height


