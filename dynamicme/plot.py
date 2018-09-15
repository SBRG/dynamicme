#============================================================
# File plot.py
# 
# Plotting functions for stress sims
#
# Laurence Yang, SBRG, UCSD
#
# 29 Aug 2017:  first version
#============================================================

import seaborn as sns
import numpy as np
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

def stacked_bar(xcol, ycol, data, hue, grp, width=0.55, by_column=True,
        ymax_fixed=None, ymin_fixed=None, poly_alpha=0.5,
        ymax_mult=1.1, debug=False, colors=None, palette_name=None, **kwargs_fc):
    """
    kwargs_fc: keyword arguments for seaborn.FacetGrid
    Note: if you get an error about infinite, try changing ymax large enough to cover
    all values.
    """
    data = data.sort_values(xcol)
    if by_column:
        g = sns.FacetGrid(data, hue=hue, col=grp, **kwargs_fc)
    else:
        g = sns.FacetGrid(data, hue=hue, row=grp, **kwargs_fc)

    cols = data[grp].unique()
    for i,ax in enumerate(g.axes.flat):        
        coli = cols[i]
        dfi = data[ data[grp]==coli]
        dfg = dfi.groupby(xcol)[ycol].sum().reset_index()
        if ymin_fixed is None:
            ymin = min(-0.1, dfi[ycol].min())
        else:
            ymin = ymin_fixed
        if ymax_fixed is None:
            ymax = dfg[ycol].max()*ymax_mult
        else:
            ymax = ymax_fixed
        if hasattr(dfi[hue],'cat'):
            hues = dfi[hue].cat.categories
        else:
            hues = dfi[hue].unique()
        if colors is None:
            if palette_name is None:
                palette_name = 'husl'
            colors = sns.color_palette(palette_name, len(hues))

        xs_all = dfi[xcol].unique().tolist()
        bottom_dict = {xi:0. for xi in xs_all}
        ymin = float(ymin)   # Need this or can error ufunc 'isfinite' not supported for the input....
        ymax = float(ymax)
        try:
            ax.set_ylim((ymin,ymax))
        except:
            print ymin,ymax
        for j,huej in enumerate(hues):
            color = colors[j]
            dfj = dfi[ dfi[hue]==huej]
            #************************************************************
            dfj = dfj.sort_values(xcol)
            #************************************************************
            xs  = dfj[xcol].values.tolist()
            xinds = [xs_all.index(xi) for xi in xs]
            ys   = dfj[ycol].tolist()
            bottoms = [bottom_dict[xi] for xi in xs]            
            try:
                #ax.bar(xinds, ys, width=width, bottom=bottoms, color=color, edgecolor='#000000', linewidth=0.5, label=huej)                

                # Since timesteps differ, widths also
                dxs = np.diff(xs)
                widths = [min(w/2., width) for w in dxs]
                widths.append(widths[-1])
                widths = [min(w1,w2) for w1,w2 in zip([width]+widths[:-1],widths)]
                ax.bar(xs, ys, width=widths, bottom=bottoms, color=color,
                        edgecolor='#000000', linewidth=0.5, label=huej)                
            except Exception as e:
                print 'i=%d. j=%d. col=%s. hue=%s.' % (i,j,coli, huej)
                print(repr(e))
                raise Exception
            #ax.set_xticks(xinds)
            #ax.set_xticklabels(xs)

            ### Show the transition lines
            polys = []
            for k in range(len(xinds)-1):
                #x1 = xinds[k]+width/2.
                #x2 = xinds[k+1]-width/2.
                x1 = xs[k] + widths[k]/2
                x2 = xs[k+1] - widths[k+1]/2

                y1 = ys[k] + bottoms[k]
                y2 = ys[k+1] + bottoms[k+1]
                xy = np.array([(x1,y1),(x1,bottoms[k]),(x2,bottoms[k+1]),(x2,y2)])
                poly = Polygon(xy, fill=True)
                polys.append(poly)                
            p = PatchCollection(polys, alpha=poly_alpha, linewidths=0.5) #, edgecolors='#000000', linewidths=0.5)
            p.set_color([color])
            ax.add_collection(p)
            ### Update bottoms
            for xk,yk in zip(xs,ys):
                bottom_dict[xk] += yk
                
        #**************************        
        if debug:                        
            for k,r in dfg.iterrows():                
                si = '%.2g'%r[ycol]
                xi = xs_all.index(r[xcol])
                yi = r[ycol]
                ax.text(xi,yi,si, ha='center', va='bottom', rotation=90)
        #**************************        
        ax.legend(bbox_to_anchor=(1,1))
        ax.set_title(coli)

    # Flip legend order
    for ax in g.axes.flat:
        handles,labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1,1))
        
    return g
