# AlphaEarth Embedding Download

Download AE embeddings from the source cooperative / TGE bucket on S3.
This code will read sub-tiles defined by lat/lon and tile height/width from the products in the bucket. It does not download the full product.

Data is stored in georeferenced `tiff` files.

Google EarthEngine authentication is required.

## Issues
If the location of interest is close to the edge of the product, it is possible that a tile is returned that does not have the location at the center.
Processing is currently limited to single products. If location +- height/width is outside the product, only the part that is contained in the product (which always includes the exact lat/lon point) is returned.
