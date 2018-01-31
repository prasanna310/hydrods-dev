# Title     : TODO
# Objective : TODO
# Created on: 1/14/18



download_daymet_nc<- function(location, start ,  end , param , path )
{
    server = "https://thredds.daac.ornl.gov/thredds/ncss/ornldaac/1328"
    print (server)
    if (length(location) != 4) {
        stop("check coordinates format: top-left / bottom-right c(lat,lon,lat,lon)")
    }
    max_year = as.numeric(format(Sys.time(), "%Y")) - 1
    if (start < 1980) {
        stop("Start year preceeds valid data range!")
    }
    if (end > max_year) {
        stop("End year exceeds valid data range!")
    }
    year_range = seq(start, end, by = 1)
    if (param == "ALL") {
        param = c("vp", "tmin", "tmax", "swe", "srad", "prcp",
            "dayl")
    }
    cat("Creating a subset of the Daymet data\n      be patient, this might take a while!\n")
    for (i in year_range) {
        for (j in param) {
            server_string = sprintf("%s/%s/daymet_v3_%s_%s_na.nc4",
                server, i, j, i)
            query = list(var = "lat", var = "lon", var = j, north = location[1],
                west = location[2], east = location[3], south = location[4],
                time_start = paste0(start, "-01-01T12:00:00Z"),
                time_end = paste0(end, "-12-30T12:00:00Z"), timeStride = 1,
                accept = "netcdf")
            daymet_file = paste0(path, "/", j, "_", i, "_ncss.nc")
            cat(paste0("Downloading DAYMET subset: ", "year: ",
                i, "; product: ", j, "\n"))
            status = try(httr::GET(url = server_string, query = query,
                httr::write_disk(path = daymet_file, overwrite = TRUE),
                httr::progress()), silent = TRUE)
            if (inherits(status, "try-error")) {
                stop("Requested coverage exceeds 6GB file size limit!")
            }
        }
    }
}

download_daymet_nc(location = c(36.61, -85.37, -85.0, 36.5), start = 1988,
    end = 1988, param = "tmin", path = ".")  #path = "~"
