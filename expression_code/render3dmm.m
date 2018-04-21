function [img, mask] = render3dmm( xIm,yIm,rgb,w,h)

%% grid construction
[x_grid,y_grid] = meshgrid(1:1:w,1:1:h);

%% Interpolate z values on the grid
Fr = scatteredInterpolant(xIm,yIm,rgb(:,1),'linear','none');
Fg = scatteredInterpolant(xIm,yIm,rgb(:,2),'linear','none');
Fb = scatteredInterpolant(xIm,yIm,rgb(:,3),'linear','none');

%% Get values for each location
imR = Fr(x_grid,y_grid);
imG = Fg(x_grid,y_grid);
imB = Fb(x_grid,y_grid);

%% Remove NaN
%  mask = isnan( imR );

%% Refine Mask for extreme posees
pts = [xIm yIm];
pts = round(pts);
pts = unique(pts,'rows');

bb = boundary(pts(:,1),pts(:,2));

mask = not(inpolygon(x_grid,y_grid,pts(bb,1),pts(bb,2)));
size(mask)

imR(mask) = 0;
imG(mask) = 0;
imB(mask) = 0;

%% building image
img(:,:,1) = imR;
img(:,:,2) = imG;
img(:,:,3) = imB;
%img = img.*255;
img = uint8(img);
end