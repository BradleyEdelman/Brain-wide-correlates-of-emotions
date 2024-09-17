function [rgbcell, rgb] = seaborn_colors(varargin)
% [rgbcell, rgb]  = seaborn_colors([num colors])
%
% 8 or 24-color HUSL Seaborn color map
% See: http://www.husl-colors.org
% https://stanford.edu/~mwaskom/software/seaborn/tutorial/color_palettes.html
% 
% Outputs:
% rgbcell : One cell per rgb color triplet
% rgb     : 3-column matrix form for rgb, as returned by colormap.m
%
% Examples:
% rgb = seaborn_colors
% rgb = seaborn_colors(8)  generate 8 colors only


rgb = [0.9677975592919913  0.44127456009157356  0.5358103155058701
    0.9699521567340649  0.4569882390259858  0.36385324448493633
    0.903599057664843  0.511987276335809  0.19588350060161624
    0.8087954113106306  0.5634700050056693  0.19502642696727285
    0.7350228985632719  0.5952719904750953  0.1944419133847522
    0.6666319352625271  0.6197366714155128  0.19396267878823373
    0.5920891529639701  0.6418467016378244  0.1935069134991043
    0.49382662140640926  0.6649121332643736  0.19300804648700284
    0.3126890019504329  0.6928754610296064  0.1923704830330379
    0.19783576093349015  0.6955516966063037  0.3995301037444499
    0.20312757197899856  0.6881249249803418  0.5177618167447304
    0.20703735729643508  0.6824290013722435  0.5885318893529169
    0.21044753832183283  0.6773105080456748  0.6433941168468681
    0.21387918628643265  0.6720135434784761  0.693961140878689
    0.21786710662428366  0.6656671601322255  0.7482809385065813
    0.22335772267769388  0.6565792317435265  0.8171355503265633
    0.23299120924703914  0.639586552066035  0.9260706093977744
    0.4768773964929644  0.5974418160509446  0.9584992622400258
    0.6423044349219739  0.5497680051256467  0.9582651433656727
    0.774710828527837  0.49133823414365724  0.9580114121137316
    0.9082572436765556  0.40195790729656516  0.9576909250290225
    0.9603888539940703  0.3814317878772117  0.8683117650835491
    0.9633321742064956  0.40643825645731757  0.7592537599568671
    0.9656056642634557  0.4245907603266889  0.6579786740552919];

n = 24;

if length(varargin) > 0
    
    % num colors entered
    n = varargin{1};
    
    if n > 24
       % error('24 colors max');
        
       while size(rgb, 1) < n
          
           m = 0.75; 
           rgb = [rgb; rgb .* m];
           m = m .* .75;
           
       end
    end
end

% choose from middle if low n, so we don't duplicate colors
len = size(rgb, 1);

wh = round(linspace(1, len * (1 + 1/len - 1/n), n));

%wh = round(linspace(1, 24, n));

rgb = rgb(wh, :);



% make cell
rgbcell = mat2cell(rgb, ones(n, 1), 3);


end % function

