<H1 align="center"> ASTR3800 Final Project Writeup </H1>
<H2 align="center"> Thor Breece </H2>
<H3 align="center"> 12/08/2022 </H3>

<H2 align="center"> Abstract </H2>

<p align="center"> Through a combination of data science, statistical analysis, machine learning and intuition, I analyzed data collected by LIGO. Throughout the project, I analyzed the Event Demographics of the LIGO data as a whole, the neutron star - neutron star merger GW170817, and discovered hidden correlations between variables within the data set. This helped me to understand the limitations of LIGO, how standard sirens, often sources of LIGO GW signals, 
are used to calculate the hubble constant, and the correlation between the Signal to Noise ratio and Luminosity Distance. </p>

<H2 align="center"> Gravitational Waves and LIGO </H2>

&nbsp;&nbsp;&nbsp;&nbsp;Gravitational waves (GWs) are ripples in gravity that are caused by the movement of massive objects such as black holes and neutron stars. Most often the gravitational waves detected are from merger events [1]. Such events can range from neutron star - neutron star mergers to black hole - black hole mergers. Gravitational waves cause a strain on an object. Causing it to either shrink or stretch based on the direction of the gravitational wave in relation to the object. Why should such a miniscule affect be studied at all? Because gravitational waves can be used to test theories of gravity and help us to understand the properties of black holes, neutron stars, and other objects in the universe. 

&nbsp;&nbsp;&nbsp;&nbsp;The Laser Interferometer Gravitational-Wave Observatory, or LIGO, is the facility that is used to detect gravitational waves. In a similar way that the oscilation of electrons to measure electro-magnetic fields, LIGO measures the dimensionless strain of mass, to detect gravitational waves [2]. A caviat of this measurement is that the dimensionless strain is only $10^{-21}$ m, which is less than the size of a proton over a kilometer. As such, detecting dimensionless strain caused by gravitational waves is extremely sensitive to enviromental changes. A truck driving by, a person walking within the facility, and even minute siesmic activity can through off the accuracy of LIGO's measurements [2]. LIGO has accounted for a number of these systematic erros in the design of its interferometer used to detect GWs. LIGO uses 2 4km arms equiped with mirrors and test masses to detect gravitational waves <i>(figure 1)</i>. Due to the extremely small change over such large distances, the longer the arms, the better chance of detecting gravitational waves.  
<figure align="center">
    <img src=interferometer.jpg width="500"> 
    <figcaption>
    <B> figure 1: </B> <cite> <a
    href = "https://www.ligo.caltech.edu/page/what-is-interferometer"> LIGO Interferometer Model 
    </a>
</figure>

<a href = "https://www.ligo.caltech.edu/page/ligos-ifo"> </a>


[1] LIGO Scientific Collaboration, B. P. Abbott et al. (2016). "Observation of Gravitational Waves from a Binary Black Hole Merger". Physical Review Letters. 116 (6): 061102. \
[2] What is an Interferometer? (n.d.). LIGO Lab | Caltech



```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from astropy import units as u
from astropy import constants as co
import linear_least_squares as lsq
from scipy.stats import pearsonr
```

<H2 align="center"> 1 Event Demographics </H2>

<H3 align="center">1.1 Redshift Limit of LIGO</H3>
&nbsp;&nbsp;&nbsp;&nbsp; The redshift limit of LIGO is the maximum stretching that a gravitational wave can recieve before it falls below the detection range of LIGO's interferometers. To determine the redshift limit of LIGO based on the data provided, plotting the redshfit of each event and applying a color map to the plot helps to better visualize the redshift of each event in the LIGO data provided (<i>figure 2</i>). It became abundantly clear that the circled point in figure 1 was the maximum detected value. While a bit of an outlier, it shows the maximum redshift LIGO was able to detect. This redshift is quantifed below the plot.


<!-- <figure align="center">
    <img src=max_redshift.png width="500"> 
    <figcaption>
    <B> figure 2: </B> Redshift for each Event in provided LIGO data
    </figcaption>
</figure> -->



```python
ligo = pd.read_csv("LIGO.csv")
event_arr = np.linspace(0,92,93)
plt.scatter(event_arr, ligo["redshift"], c=ligo["redshift"], cmap="Reds")
cbar = plt.colorbar()
cbar.set_label("Redshift")
plt.xlabel("# Event (in order of data)")
plt.ylabel("Redshift")
plt.title("figure 2: Redshift for each Event in provided LIGO data")
fig = plt.gcf()
ax = fig.gca()
e = Ellipse((ligo["redshift"].argmax(), ligo["redshift"].max()), 5, 0.2, fc='None', ec='cyan')
ax.add_patch(e);


```


    
![png](README_files/README_7_0.png)
    



```python
print(f'The circled data point in the above plot represents the maximum redshift in the given dataset. \nFinding this value within the data, the redshift limit is {ligo["redshift"].max()}' )

```

    The circled data point in the above plot represents the maximum redshift in the given dataset. 
    Finding this value within the data, the redshift limit is 1.18


<H3 align="center">1.2 Relationship between binary mass and distance</H3> 


&nbsp;&nbsp;&nbsp;&nbsp; To determine if the binary mass is correlated to the distance, plotting the total mass of the system versus luminosity distance is helpful (<i>figure 3</i>). Inuitively it appears to have a linear correlation so the next step was to perform a linear least squares curve fit of the data to determine if it fit a model. The red line represents the model and the green lines represent 1 $\sigma$ of the data. As can be seen, the majority of the data falls within 1 $\sigma$ of the model. This provides compelling evidence that a correlation exists, but is not strong enough to come to a conclusion.


```python
ligo = ligo[ligo['luminosity_distance'].notna()]
ligo = ligo[ligo['total_mass_source'].notna()]
plt.scatter(ligo["luminosity_distance"], ligo["total_mass_source"])
plt.xlabel("Distance (Mpc)")
plt.ylabel(r"Mass ($M_{\odot}$)");
uslope = lsq.m_wtd(ligo['luminosity_distance'], ligo['total_mass_source'], ligo['total_mass_source_upper'])
uintercept = lsq.b_wtd(ligo['luminosity_distance'], ligo['total_mass_source'], ligo['total_mass_source_upper'])
model = uslope * ligo["luminosity_distance"] + uintercept
div = ligo["total_mass_source"].std()
plt.plot(ligo["luminosity_distance"], model+div, color="g", label=f'y={uslope:.2f}x+{uintercept:.2f}+$1\sigma$')
plt.plot(ligo["luminosity_distance"], model, color="r", label=f'y={uslope:.2f}x+{uintercept:.2f}')
plt.plot(ligo["luminosity_distance"], model-div, color="g", label=f'y={uslope:.2f}x+{uintercept:.2f}-$1\sigma$')
plt.title("Figure 1: Mass of Binary System versus Distance")
plt.legend();
```


    
![png](README_files/README_11_0.png)
    


To determine the strength of the signal, the next step is to compute the correlation coefficeint of the two variables. 


```python
def cor(x,y):
    mean_x = np.average(x)
    mean_y = np.average(y)
    numerator = np.sum((x - mean_x)*(y-mean_y))
    denominator = np.sqrt(np.sum((x-mean_x)**2) * np.sum((y-mean_y)**2))
    return numerator / denominator
print(f'Correlation coefficient: {cor(ligo["total_mass_source"],ligo["luminosity_distance"]):.2f}')
```

    Correlation coefficient: 0.67



<!-- ```
def cor(x,y):
    mean_x = np.average(x)
    mean_y = np.average(y)
    numerator = np.sum((x - mean_x)*(y-mean_y))
    denominator = np.sqrt(np.sum((x-mean_x)**2) * np.sum((y-mean_y)**2))
    return numerator / denominator
```  -->
This correlation coefficient shows that the two varialbes have a reasonabaly significant correlation. To determine the likelyhood that such a correlation happened by chance, running a pearsonr calculation to determine the probability of finding such a correlation in uncorrelated data is a clear next step.

<!-- <figure align="center">
    <img src=mass_lumen.png width="500"> 
    <figcaption>
    <B> figure 3: </B> The relationship between the mass of a binary system and its luminosity distance with a model, with standard deviation, representing said relationship.
    </figcaption>
</figure> -->


```python
r, p = pearsonr(ligo["total_mass_source"],ligo["luminosity_distance"])
print(f'p = {p:.2g}')
```

    p = 1.9e-11


Given that the pearson probability was extremely low, it is highly likely that the correlation did not occur merely by chance. The significance of this is that detectable GWs from deep space binary systems have to come from massive objects, otherwise the signal is too faint for LIGO to detect - being lost in the noise or redshifted past LIGO's detection limit.

<H3 align="center">1.3 Mass to Gravitational Wave conversion </H3>

&nbsp;&nbsp;&nbsp;&nbsp; The amount of mass lost from a given merger is easily calculated from the difference of the total mass prior to the merger and the final mass after the merger.
$$
m_{T} - m_{F} = m_{\Delta}
$$
The fraction of the original mass coverted into GWs can be calculated as
$$
m_{frac} = 1 - \frac{m_{T}}{m_{F}}
$$

<!-- <figure align="center">
    <img src=energy.png width="500"> 
    <figcaption>
    <B> figure 3: </B> The Energy released from each merger event.
    </figcaption>
</figure> -->


```python
ligo = ligo[ligo["total_mass_source"].notna()]
ligo = ligo[ligo["final_mass_source"].notna()]
massToGW = ligo["total_mass_source"] - ligo['final_mass_source']
mass_frac = 1 - (ligo["final_mass_source"] /ligo["total_mass_source"])
mass_delta = list(ligo["total_mass_source"] - ligo["final_mass_source"])
mass_frac_avg = np.average(mass_frac)
mass_frac_std = np.std(mass_frac)
print(f'The average fraction of mass converted to GWs is {mass_frac_avg:.3f} and the spread, or standard deviation in this fraction is {mass_frac_std:.5f}.')

```

    The average fraction of mass converted to GWs is 0.043 and the spread, or standard deviation in this fraction is 0.01035.


<H3 align="center">1.4 Gravitational Wave Luminosities</H3>


What happens to this fraction of mass? It is converted back in to energy according to the famous equation $ E = mc^{2}$. As shown in <i>figure 4</i>, this equates to massive amounts of energy for each of the mergers in the data. Unsuprisingly, it looks very similar to the redshift plot. As the redshift (by extention of the fact that distance is correlated with mass) was determined to be correlated to mass in part 1.2, it makes sense that the Energy would be correlated to redshift as well.

Before this calculation, however, it was necessary to define an inband peak duration. From tutorial 12 part 1.9 we get an inband peak of 63 ms, which is the assumption for this calculation of GW luminosity.


```python
m_tot = ligo["total_mass_source"].to_numpy()
m_fin = ligo["final_mass_source"].to_numpy()
dM = (m_tot - m_fin) * u.M_sun
E = dM * co.c ** 2
E = [i.decompose() for i in E]
duration = 0.063 * u.second
gwL = [(i / duration).to('L_sun') for i in E]
gwL_plotting = [i.value for i in gwL]
# print("from tutorial 12 part 1.9 we get a inband peak duration which I will be using as the assumption for the caluclation of GW luminosity")
plt.scatter(range(len(gwL)), gwL_plotting)
plt.title("figure 4: GWL for each event in LIGO data")
plt.xlabel("# event")
plt.ylabel(r"Luminosity ($L_{\odot}$)");

```


    
![png](README_files/README_22_0.png)
    



```python
print(f'The spread in the GW Luminosity is {np.array(gwL_plotting).std():.2e} Solar Luminosities.')
supernovae = 570000000000
ratio = np.array(gwL_plotting).max() / supernovae
print(f'Comparing the most luminous event with a luminosity of {np.array(gwL_plotting).max():.2e} solar luminosities to the luminosity of the most luminous supernovae, ASASSN-15lh, which is {supernovae:.2e} Solar Luminosities [3]. It is {ratio:.2e} times more luminous.')
# SOURCE GIA CATALOG
```

    The spread in the GW Luminosity is 1.24e+22 Solar Luminosities.
    Comparing the most luminous event with a luminosity of 6.97e+22 solar luminosities to the luminosity of the most luminous supernovae, ASASSN-15lh, which is 5.70e+11 Solar Luminosities [3]. It is 1.22e+11 times more luminous.


[3] Dong, Subo; et al. (2015). "ASASSN-15lh: A highly super-luminous supernova". Science. 351 (6276): 257–260.

<H3 align="center">1.5 Binary Mass Ratio</H3>

While this did not result in an obvious distribution of the binary mass ratio, <i>figure 5</i> shows the linear relationship between the masses of the two objects in a binary system.


```python
plt.scatter(ligo["mass_1_source"], ligo["mass_2_source"])
plt.title("Mass 1 vs Mass 2")
plt.xlabel(r'Mass $M_{\odot}$')
plt.ylabel(r'Mass $M_{\odot}$');

```


    
![png](README_files/README_27_0.png)
    


By plotting a histogram of the ratio of mass 1 to mass 2 it becomes clear that the most common binary mass ratio sits within the range of 1 to 5 (<i>figure 6</i>). 


```python
b_mass_r = ligo["mass_1_source"] / ligo["mass_2_source"]
plt.hist(b_mass_r, bins=50,edgecolor='black')
plt.xlabel("Binary Mass Ratio")
plt.ylabel("Number of occurances")
plt.title("Histogram of most frequent Binary Mass ratios");
```


    
![png](README_files/README_29_0.png)
    



```python
print(f'The minimum binary mass ratio is {b_mass_r.min():.2f} and the maximum binary mass ratio is {b_mass_r.max():.2f}')
```

    The minimum binary mass ratio is 1.19 and the maximum binary mass ratio is 26.58


<H2 align="center"> 2 Standard Sirens </H2>

<H3 align="center">2.1 GW170817: a neutron star - neutron star merger </H3>

GW170817 is a notable merger as it can be used as a "standard siren" thus it can be used to measure the Hubble constant with little information than it's luminosity distance and host galaxy's recession velocity.


```python
ligo = pd.read_csv("LIGO.csv")
GW170817 = ligo.loc[ligo["commonName"] == "GW170817"]
dl = float(GW170817["luminosity_distance"].to_string(index=False))
ldu = float(GW170817["luminosity_distance_upper"].to_string(index=False))
ldl = float(GW170817["luminosity_distance_lower"].to_string(index=False))
print(f'The luminosity distance of this merger is {dl} Mpc with an uncertainty of + {ldu} and {ldl} Mpc.')
```

    The luminosity distance of this merger is 40.0 Mpc with an uncertainty of + 7.0 and -15.0 Mpc.


<H3 align="center">2.2 Hubble constant calculation</H3>

To determine the hubble constant from the Luminosity Distance $D_L$ and the recesion velocity $v$:
$$
cz = H_0 D_L
$$
To avoid confusion and to simplify the problem, the equation for redshift was plugged into the equation. This is an extra step but helps clarify the relationship: $z = \frac{v}{c} $
$$
cz = H_0D_L \\
H_0 = \frac{cz}{D_L} \\
\text{after plugging in the equation for redshift:} \\ 
H_0 = \frac{v}{D_L} 
$$
Computing the Hubble cosntant based on GW170817 in python:


```python
v = 3017

H_0 = v / dl
print(f'The Hubble Constant as calculated via this merger is {H_0:.2f} km/s/mpc ')
```

    The Hubble Constant as calculated via this merger is 75.42 km/s/mpc 


This value is close to the current calculation of 74 km/s/Mpc.

<H3 align="center">2.3 Uncertainty on the Hubble constant</H3>

To calculate the uncertainty on the Hubble cosntant, error propagation must be performed on both the Luminosity distance and the velocity. To perform error propogation the formula for a multivariable equation is:
$$
\sigma^{2}_f = (\frac{\partial f}{\partial x})^{2} \sigma^{2}_x + (\frac{\partial f}{\partial y})^{2} \sigma^{2}_y + ...
$$
For this instance: 
$$
\sigma^{2}_{H_{0}} = (\frac{\partial}{\partial v}\frac{v}{D_L})^{2}\sigma_{v}^2 + (\frac{\partial}{\partial D_L}\frac{v}{D_L})^{2}\sigma_{D_L}^2 \\
\sigma_{H_0} = \sqrt{(\frac{\partial}{\partial v}\frac{v}{D_L})^{2}\sigma_{v}^2 + (\frac{\partial}{\partial D_L}\frac{v}{D_L})^{2}\sigma_{D_L}^2 } \\
\sigma_{H_0} = \sqrt{(\frac{1}{D_L})^2 \sigma_{v}^2 + (\frac{-v}{(D_L)^2})^2 \sigma_{D_L}^2} \\
 \sigma_{H_0} = \sqrt{(D_L)^{-2}\sigma_{v}^2 + \frac{v^{2}}{(D_L)^4}\sigma_{D_L}^2} \\
$$
As there is an asymetric uncertainty on $D_L$, error propegation is done using the upper, lower and the average uncertainty of $D_L$.


```python
sig_v = 166
sigma_H_u = np.sqrt(dl**-2 * sig_v**2 + (3017**2 / (dl)**4) * ldu ** 2)
sigma_H_l = np.sqrt(dl**-2 * sig_v**2 + (3017**2 / (dl)**4) * ldl ** 2)
sigma_H_a = np.sqrt(dl**-2 * sig_v**2 + (3017**2 / (dl)**4) * 11 ** 2)
print(f'Uncertainty using upper bound of Luminosity Distance {sigma_H_u:.2f}')
print(f'Uncertainty using lower bound of Luminosity Distance {sigma_H_l:.2f}')
print(f'Uncertainty using average bound of Luminosity Distance {sigma_H_a:.2f}')
```

    Uncertainty using upper bound of Luminosity Distance 13.84
    Uncertainty using lower bound of Luminosity Distance 28.59
    Uncertainty using average bound of Luminosity Distance 21.15


The error based on this calculation is way off the mark compared to the error of $\pm$ 2.4 km/s/mpc. To reduce the error within 10% we would need at hundreds of similar events and to reduce the error within 1% we would need thousands if not tens of thousands of similar events.

<H3 align="center">3 Finding Hidden Correlations</H3>

There are many correlations between the variables within the LIGO data. Soem of them ore obvious ones are how the final mass goes up perfectly linearly with the total mass. Similarly as obvious, as they are directly calculated from one another, redshift and luminosity distance show a indisputable correlation. What is more intriguing are the not obvious correlations and the suprising and confusing correlations. The objective not only being to discover hidden connections between variables but to better understand the data as a whole.  

Firstly, a correlation matrix was generated generate between all variables after some cleaning up of the data was done. Due to a large number of missing values and a number of columns that seem likely to produce errors. These columns were dropped from the dataset. 


```python
pd.set_option("display.max_columns", None)

```


```python
ligo = pd.read_csv("LIGO.csv")
cols = list(ligo.columns)
ligo.fillna(0, inplace=True)

quant = ligo

quant.drop(['id', 'commonName', 'version', 'catalog.shortName', 'GPS', 'reference', 'jsonurl'], axis = 1, inplace=True)
cols = list(quant.columns)
```


```python
corrr = quant.corr()
droppable = ["chirp_mass", "chirp_mass_lower", "chirp_mass_upper", "far_lower", "far_upper", "p_astro_upper", "p_astro_lower"]
corr = corrr.drop(droppable, axis = 1)

cor = corr.dropna()
print(f"Due to NaN occuring in a fair number of the upper and lower bounds on some variables, \nThese {droppable} \nwere dropped from the correlation matrix")
cor
```

    Due to NaN occuring in a fair number of the upper and lower bounds on some variables, 
    These ['chirp_mass', 'chirp_mass_lower', 'chirp_mass_upper', 'far_lower', 'far_upper', 'p_astro_upper', 'p_astro_lower'] 
    were dropped from the correlation matrix





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mass_1_source</th>
      <th>mass_1_source_lower</th>
      <th>mass_1_source_upper</th>
      <th>mass_2_source</th>
      <th>mass_2_source_lower</th>
      <th>mass_2_source_upper</th>
      <th>network_matched_filter_snr</th>
      <th>network_matched_filter_snr_lower</th>
      <th>network_matched_filter_snr_upper</th>
      <th>luminosity_distance</th>
      <th>luminosity_distance_lower</th>
      <th>luminosity_distance_upper</th>
      <th>chi_eff</th>
      <th>chi_eff_lower</th>
      <th>chi_eff_upper</th>
      <th>total_mass_source</th>
      <th>total_mass_source_lower</th>
      <th>total_mass_source_upper</th>
      <th>chirp_mass_source</th>
      <th>chirp_mass_source_lower</th>
      <th>chirp_mass_source_upper</th>
      <th>redshift</th>
      <th>redshift_lower</th>
      <th>redshift_upper</th>
      <th>far</th>
      <th>p_astro</th>
      <th>final_mass_source</th>
      <th>final_mass_source_lower</th>
      <th>final_mass_source_upper</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mass_1_source</th>
      <td>1.000000</td>
      <td>-0.836269</td>
      <td>0.562270</td>
      <td>0.887676</td>
      <td>-0.900422</td>
      <td>0.922229</td>
      <td>-0.110595</td>
      <td>-0.257566</td>
      <td>0.258132</td>
      <td>0.730336</td>
      <td>-0.712231</td>
      <td>0.662752</td>
      <td>0.205470</td>
      <td>-0.676580</td>
      <td>0.601442</td>
      <td>0.885688</td>
      <td>-0.812106</td>
      <td>0.640544</td>
      <td>0.949259</td>
      <td>-0.934089</td>
      <td>0.884363</td>
      <td>0.745512</td>
      <td>-0.725662</td>
      <td>0.673544</td>
      <td>-0.007988</td>
      <td>0.093121</td>
      <td>0.983488</td>
      <td>-0.849534</td>
      <td>0.630959</td>
    </tr>
    <tr>
      <th>mass_1_source_lower</th>
      <td>-0.836269</td>
      <td>1.000000</td>
      <td>-0.828535</td>
      <td>-0.568071</td>
      <td>0.711730</td>
      <td>-0.898225</td>
      <td>0.349217</td>
      <td>0.547325</td>
      <td>-0.588031</td>
      <td>-0.804140</td>
      <td>0.806657</td>
      <td>-0.866988</td>
      <td>-0.412397</td>
      <td>0.756592</td>
      <td>-0.691879</td>
      <td>-0.729642</td>
      <td>0.900573</td>
      <td>-0.850773</td>
      <td>-0.678199</td>
      <td>0.822254</td>
      <td>-0.888634</td>
      <td>-0.802016</td>
      <td>0.808912</td>
      <td>-0.866802</td>
      <td>-0.213490</td>
      <td>0.153595</td>
      <td>-0.758022</td>
      <td>0.937873</td>
      <td>-0.854995</td>
    </tr>
    <tr>
      <th>mass_1_source_upper</th>
      <td>0.562270</td>
      <td>-0.828535</td>
      <td>1.000000</td>
      <td>0.359750</td>
      <td>-0.532617</td>
      <td>0.636376</td>
      <td>-0.344238</td>
      <td>-0.542651</td>
      <td>0.702981</td>
      <td>0.593601</td>
      <td>-0.604560</td>
      <td>0.746210</td>
      <td>0.379709</td>
      <td>-0.634561</td>
      <td>0.690389</td>
      <td>0.489191</td>
      <td>-0.770688</td>
      <td>0.949197</td>
      <td>0.435975</td>
      <td>-0.602075</td>
      <td>0.716073</td>
      <td>0.600227</td>
      <td>-0.623515</td>
      <td>0.754730</td>
      <td>0.284003</td>
      <td>-0.176180</td>
      <td>0.505128</td>
      <td>-0.804109</td>
      <td>0.977386</td>
    </tr>
    <tr>
      <th>mass_2_source</th>
      <td>0.887676</td>
      <td>-0.568071</td>
      <td>0.359750</td>
      <td>1.000000</td>
      <td>-0.911964</td>
      <td>0.779176</td>
      <td>0.024964</td>
      <td>-0.008197</td>
      <td>0.023920</td>
      <td>0.570803</td>
      <td>-0.548763</td>
      <td>0.443756</td>
      <td>-0.005803</td>
      <td>-0.542892</td>
      <td>0.514833</td>
      <td>0.824987</td>
      <td>-0.653423</td>
      <td>0.450468</td>
      <td>0.985843</td>
      <td>-0.863293</td>
      <td>0.760073</td>
      <td>0.599667</td>
      <td>-0.576396</td>
      <td>0.463283</td>
      <td>-0.086445</td>
      <td>0.222898</td>
      <td>0.955395</td>
      <td>-0.683857</td>
      <td>0.437540</td>
    </tr>
    <tr>
      <th>mass_2_source_lower</th>
      <td>-0.900422</td>
      <td>0.711730</td>
      <td>-0.532617</td>
      <td>-0.911964</td>
      <td>1.000000</td>
      <td>-0.895529</td>
      <td>0.151433</td>
      <td>0.150079</td>
      <td>-0.182771</td>
      <td>-0.646336</td>
      <td>0.646572</td>
      <td>-0.595655</td>
      <td>-0.083591</td>
      <td>0.634410</td>
      <td>-0.650846</td>
      <td>-0.868117</td>
      <td>0.789742</td>
      <td>-0.607767</td>
      <td>-0.932792</td>
      <td>0.967149</td>
      <td>-0.884494</td>
      <td>-0.666397</td>
      <td>0.668586</td>
      <td>-0.610215</td>
      <td>-0.019749</td>
      <td>-0.049260</td>
      <td>-0.930379</td>
      <td>0.805391</td>
      <td>-0.589172</td>
    </tr>
    <tr>
      <th>mass_2_source_upper</th>
      <td>0.922229</td>
      <td>-0.898225</td>
      <td>0.636376</td>
      <td>0.779176</td>
      <td>-0.895529</td>
      <td>1.000000</td>
      <td>-0.280728</td>
      <td>-0.361336</td>
      <td>0.389918</td>
      <td>0.815483</td>
      <td>-0.826858</td>
      <td>0.827404</td>
      <td>0.249988</td>
      <td>-0.736659</td>
      <td>0.685756</td>
      <td>0.854640</td>
      <td>-0.895920</td>
      <td>0.712259</td>
      <td>0.853188</td>
      <td>-0.954304</td>
      <td>0.975359</td>
      <td>0.819218</td>
      <td>-0.833242</td>
      <td>0.830086</td>
      <td>0.177403</td>
      <td>-0.075916</td>
      <td>0.895344</td>
      <td>-0.919999</td>
      <td>0.696836</td>
    </tr>
    <tr>
      <th>network_matched_filter_snr</th>
      <td>-0.110595</td>
      <td>0.349217</td>
      <td>-0.344238</td>
      <td>0.024964</td>
      <td>0.151433</td>
      <td>-0.280728</td>
      <td>1.000000</td>
      <td>0.581301</td>
      <td>-0.441130</td>
      <td>-0.455209</td>
      <td>0.461442</td>
      <td>-0.469594</td>
      <td>-0.161008</td>
      <td>0.450793</td>
      <td>-0.448745</td>
      <td>-0.130492</td>
      <td>0.347640</td>
      <td>-0.333382</td>
      <td>-0.026059</td>
      <td>0.227052</td>
      <td>-0.303416</td>
      <td>-0.463817</td>
      <td>0.474338</td>
      <td>-0.484218</td>
      <td>-0.224901</td>
      <td>0.387586</td>
      <td>-0.064574</td>
      <td>0.352203</td>
      <td>-0.340329</td>
    </tr>
    <tr>
      <th>network_matched_filter_snr_lower</th>
      <td>-0.257566</td>
      <td>0.547325</td>
      <td>-0.542651</td>
      <td>-0.008197</td>
      <td>0.150079</td>
      <td>-0.361336</td>
      <td>0.581301</td>
      <td>1.000000</td>
      <td>-0.817297</td>
      <td>-0.487856</td>
      <td>0.489776</td>
      <td>-0.623510</td>
      <td>-0.231526</td>
      <td>0.585364</td>
      <td>-0.548700</td>
      <td>-0.296407</td>
      <td>0.542369</td>
      <td>-0.561993</td>
      <td>-0.087076</td>
      <td>0.253999</td>
      <td>-0.394109</td>
      <td>-0.493800</td>
      <td>0.499734</td>
      <td>-0.638689</td>
      <td>-0.357351</td>
      <td>0.150394</td>
      <td>-0.175326</td>
      <td>0.514041</td>
      <td>-0.527470</td>
    </tr>
    <tr>
      <th>network_matched_filter_snr_upper</th>
      <td>0.258132</td>
      <td>-0.588031</td>
      <td>0.702981</td>
      <td>0.023920</td>
      <td>-0.182771</td>
      <td>0.389918</td>
      <td>-0.441130</td>
      <td>-0.817297</td>
      <td>1.000000</td>
      <td>0.426937</td>
      <td>-0.440702</td>
      <td>0.676960</td>
      <td>0.261816</td>
      <td>-0.588897</td>
      <td>0.581646</td>
      <td>0.275318</td>
      <td>-0.594578</td>
      <td>0.673226</td>
      <td>0.094421</td>
      <td>-0.266956</td>
      <td>0.465225</td>
      <td>0.435492</td>
      <td>-0.461435</td>
      <td>0.691294</td>
      <td>0.601730</td>
      <td>-0.182484</td>
      <td>0.186712</td>
      <td>-0.582241</td>
      <td>0.662470</td>
    </tr>
    <tr>
      <th>luminosity_distance</th>
      <td>0.730336</td>
      <td>-0.804140</td>
      <td>0.593601</td>
      <td>0.570803</td>
      <td>-0.646336</td>
      <td>0.815483</td>
      <td>-0.455209</td>
      <td>-0.487856</td>
      <td>0.426937</td>
      <td>1.000000</td>
      <td>-0.989751</td>
      <td>0.895056</td>
      <td>0.417430</td>
      <td>-0.727114</td>
      <td>0.577861</td>
      <td>0.698823</td>
      <td>-0.795239</td>
      <td>0.667054</td>
      <td>0.643064</td>
      <td>-0.746192</td>
      <td>0.819701</td>
      <td>0.995870</td>
      <td>-0.983113</td>
      <td>0.886440</td>
      <td>0.138640</td>
      <td>-0.100031</td>
      <td>0.690123</td>
      <td>-0.808712</td>
      <td>0.650764</td>
    </tr>
    <tr>
      <th>luminosity_distance_lower</th>
      <td>-0.712231</td>
      <td>0.806657</td>
      <td>-0.604560</td>
      <td>-0.548763</td>
      <td>0.646572</td>
      <td>-0.826858</td>
      <td>0.461442</td>
      <td>0.489776</td>
      <td>-0.440702</td>
      <td>-0.989751</td>
      <td>1.000000</td>
      <td>-0.916128</td>
      <td>-0.411644</td>
      <td>0.728159</td>
      <td>-0.585533</td>
      <td>-0.681533</td>
      <td>0.803671</td>
      <td>-0.677099</td>
      <td>-0.621894</td>
      <td>0.751012</td>
      <td>-0.836249</td>
      <td>-0.983032</td>
      <td>0.994102</td>
      <td>-0.907620</td>
      <td>-0.175960</td>
      <td>0.141725</td>
      <td>-0.670720</td>
      <td>0.817090</td>
      <td>-0.662247</td>
    </tr>
    <tr>
      <th>luminosity_distance_upper</th>
      <td>0.662752</td>
      <td>-0.866988</td>
      <td>0.746210</td>
      <td>0.443756</td>
      <td>-0.595655</td>
      <td>0.827404</td>
      <td>-0.469594</td>
      <td>-0.623510</td>
      <td>0.676960</td>
      <td>0.895056</td>
      <td>-0.916128</td>
      <td>1.000000</td>
      <td>0.428542</td>
      <td>-0.757330</td>
      <td>0.650912</td>
      <td>0.628941</td>
      <td>-0.870432</td>
      <td>0.774683</td>
      <td>0.529369</td>
      <td>-0.706634</td>
      <td>0.861952</td>
      <td>0.887721</td>
      <td>-0.914158</td>
      <td>0.995767</td>
      <td>0.458951</td>
      <td>-0.254961</td>
      <td>0.604935</td>
      <td>-0.884885</td>
      <td>0.767169</td>
    </tr>
    <tr>
      <th>chi_eff</th>
      <td>0.205470</td>
      <td>-0.412397</td>
      <td>0.379709</td>
      <td>-0.005803</td>
      <td>-0.083591</td>
      <td>0.249988</td>
      <td>-0.161008</td>
      <td>-0.231526</td>
      <td>0.261816</td>
      <td>0.417430</td>
      <td>-0.411644</td>
      <td>0.428542</td>
      <td>1.000000</td>
      <td>-0.124103</td>
      <td>0.049012</td>
      <td>0.114804</td>
      <td>-0.280149</td>
      <td>0.284239</td>
      <td>0.075491</td>
      <td>-0.167288</td>
      <td>0.221006</td>
      <td>0.394462</td>
      <td>-0.391843</td>
      <td>0.409495</td>
      <td>0.118732</td>
      <td>-0.078059</td>
      <td>0.125551</td>
      <td>-0.318183</td>
      <td>0.318159</td>
    </tr>
    <tr>
      <th>chi_eff_lower</th>
      <td>-0.676580</td>
      <td>0.756592</td>
      <td>-0.634561</td>
      <td>-0.542892</td>
      <td>0.634410</td>
      <td>-0.736659</td>
      <td>0.450793</td>
      <td>0.585364</td>
      <td>-0.588897</td>
      <td>-0.727114</td>
      <td>0.728159</td>
      <td>-0.757330</td>
      <td>-0.124103</td>
      <td>1.000000</td>
      <td>-0.794864</td>
      <td>-0.627852</td>
      <td>0.748048</td>
      <td>-0.658111</td>
      <td>-0.603235</td>
      <td>0.686198</td>
      <td>-0.744060</td>
      <td>-0.749008</td>
      <td>0.755424</td>
      <td>-0.778233</td>
      <td>-0.287039</td>
      <td>0.008903</td>
      <td>-0.650182</td>
      <td>0.772704</td>
      <td>-0.658188</td>
    </tr>
    <tr>
      <th>chi_eff_upper</th>
      <td>0.601442</td>
      <td>-0.691879</td>
      <td>0.690389</td>
      <td>0.514833</td>
      <td>-0.650846</td>
      <td>0.685756</td>
      <td>-0.448745</td>
      <td>-0.548700</td>
      <td>0.581646</td>
      <td>0.577861</td>
      <td>-0.585533</td>
      <td>0.650912</td>
      <td>0.049012</td>
      <td>-0.794864</td>
      <td>1.000000</td>
      <td>0.573547</td>
      <td>-0.707605</td>
      <td>0.699102</td>
      <td>0.560339</td>
      <td>-0.657482</td>
      <td>0.709686</td>
      <td>0.609212</td>
      <td>-0.626352</td>
      <td>0.684676</td>
      <td>0.266861</td>
      <td>0.064235</td>
      <td>0.591684</td>
      <td>-0.727054</td>
      <td>0.710832</td>
    </tr>
    <tr>
      <th>total_mass_source</th>
      <td>0.885688</td>
      <td>-0.729642</td>
      <td>0.489191</td>
      <td>0.824987</td>
      <td>-0.868117</td>
      <td>0.854640</td>
      <td>-0.130492</td>
      <td>-0.296407</td>
      <td>0.275318</td>
      <td>0.698823</td>
      <td>-0.681533</td>
      <td>0.628941</td>
      <td>0.114804</td>
      <td>-0.627852</td>
      <td>0.573547</td>
      <td>1.000000</td>
      <td>-0.852864</td>
      <td>0.649287</td>
      <td>0.866708</td>
      <td>-0.892172</td>
      <td>0.834537</td>
      <td>0.718334</td>
      <td>-0.699599</td>
      <td>0.644463</td>
      <td>0.013607</td>
      <td>0.051566</td>
      <td>0.886376</td>
      <td>-0.792724</td>
      <td>0.569174</td>
    </tr>
    <tr>
      <th>total_mass_source_lower</th>
      <td>-0.812106</td>
      <td>0.900573</td>
      <td>-0.770688</td>
      <td>-0.653423</td>
      <td>0.789742</td>
      <td>-0.895920</td>
      <td>0.347640</td>
      <td>0.542369</td>
      <td>-0.594578</td>
      <td>-0.795239</td>
      <td>0.803671</td>
      <td>-0.870432</td>
      <td>-0.280149</td>
      <td>0.748048</td>
      <td>-0.707605</td>
      <td>-0.852864</td>
      <td>1.000000</td>
      <td>-0.854002</td>
      <td>-0.721092</td>
      <td>0.872022</td>
      <td>-0.925564</td>
      <td>-0.798800</td>
      <td>0.812026</td>
      <td>-0.876042</td>
      <td>-0.312846</td>
      <td>0.193786</td>
      <td>-0.779342</td>
      <td>0.974899</td>
      <td>-0.810728</td>
    </tr>
    <tr>
      <th>total_mass_source_upper</th>
      <td>0.640544</td>
      <td>-0.850773</td>
      <td>0.949197</td>
      <td>0.450468</td>
      <td>-0.607767</td>
      <td>0.712259</td>
      <td>-0.333382</td>
      <td>-0.561993</td>
      <td>0.673226</td>
      <td>0.667054</td>
      <td>-0.677099</td>
      <td>0.774683</td>
      <td>0.284239</td>
      <td>-0.658111</td>
      <td>0.699102</td>
      <td>0.649287</td>
      <td>-0.854002</td>
      <td>1.000000</td>
      <td>0.524690</td>
      <td>-0.697204</td>
      <td>0.796801</td>
      <td>0.671832</td>
      <td>-0.691496</td>
      <td>0.782872</td>
      <td>0.212604</td>
      <td>-0.191058</td>
      <td>0.587551</td>
      <td>-0.843652</td>
      <td>0.983461</td>
    </tr>
    <tr>
      <th>chirp_mass_source</th>
      <td>0.949259</td>
      <td>-0.678199</td>
      <td>0.435975</td>
      <td>0.985843</td>
      <td>-0.932792</td>
      <td>0.853188</td>
      <td>-0.026059</td>
      <td>-0.087076</td>
      <td>0.094421</td>
      <td>0.643064</td>
      <td>-0.621894</td>
      <td>0.529369</td>
      <td>0.075491</td>
      <td>-0.603235</td>
      <td>0.560339</td>
      <td>0.866708</td>
      <td>-0.721092</td>
      <td>0.524690</td>
      <td>1.000000</td>
      <td>-0.913305</td>
      <td>0.822390</td>
      <td>0.668350</td>
      <td>-0.645755</td>
      <td>0.547030</td>
      <td>-0.076844</td>
      <td>0.188736</td>
      <td>0.988833</td>
      <td>-0.754979</td>
      <td>0.512854</td>
    </tr>
    <tr>
      <th>chirp_mass_source_lower</th>
      <td>-0.934089</td>
      <td>0.822254</td>
      <td>-0.602075</td>
      <td>-0.863293</td>
      <td>0.967149</td>
      <td>-0.954304</td>
      <td>0.227052</td>
      <td>0.253999</td>
      <td>-0.266956</td>
      <td>-0.746192</td>
      <td>0.751012</td>
      <td>-0.706634</td>
      <td>-0.167288</td>
      <td>0.686198</td>
      <td>-0.657482</td>
      <td>-0.892172</td>
      <td>0.872022</td>
      <td>-0.697204</td>
      <td>-0.913305</td>
      <td>1.000000</td>
      <td>-0.942177</td>
      <td>-0.756599</td>
      <td>0.763632</td>
      <td>-0.716411</td>
      <td>-0.023585</td>
      <td>0.046610</td>
      <td>-0.932274</td>
      <td>0.886512</td>
      <td>-0.674015</td>
    </tr>
    <tr>
      <th>chirp_mass_source_upper</th>
      <td>0.884363</td>
      <td>-0.888634</td>
      <td>0.716073</td>
      <td>0.760073</td>
      <td>-0.884494</td>
      <td>0.975359</td>
      <td>-0.303416</td>
      <td>-0.394109</td>
      <td>0.465225</td>
      <td>0.819701</td>
      <td>-0.836249</td>
      <td>0.861952</td>
      <td>0.221006</td>
      <td>-0.744060</td>
      <td>0.709686</td>
      <td>0.834537</td>
      <td>-0.925564</td>
      <td>0.796801</td>
      <td>0.822390</td>
      <td>-0.942177</td>
      <td>1.000000</td>
      <td>0.823900</td>
      <td>-0.844808</td>
      <td>0.865053</td>
      <td>0.250440</td>
      <td>-0.132329</td>
      <td>0.865212</td>
      <td>-0.946562</td>
      <td>0.782916</td>
    </tr>
    <tr>
      <th>redshift</th>
      <td>0.745512</td>
      <td>-0.802016</td>
      <td>0.600227</td>
      <td>0.599667</td>
      <td>-0.666397</td>
      <td>0.819218</td>
      <td>-0.463817</td>
      <td>-0.493800</td>
      <td>0.435492</td>
      <td>0.995870</td>
      <td>-0.983032</td>
      <td>0.887721</td>
      <td>0.394462</td>
      <td>-0.749008</td>
      <td>0.609212</td>
      <td>0.718334</td>
      <td>-0.798800</td>
      <td>0.671832</td>
      <td>0.668350</td>
      <td>-0.756599</td>
      <td>0.823900</td>
      <td>1.000000</td>
      <td>-0.985306</td>
      <td>0.886252</td>
      <td>0.139046</td>
      <td>-0.062069</td>
      <td>0.711503</td>
      <td>-0.812215</td>
      <td>0.655919</td>
    </tr>
    <tr>
      <th>redshift_lower</th>
      <td>-0.725662</td>
      <td>0.808912</td>
      <td>-0.623515</td>
      <td>-0.576396</td>
      <td>0.668586</td>
      <td>-0.833242</td>
      <td>0.474338</td>
      <td>0.499734</td>
      <td>-0.461435</td>
      <td>-0.983113</td>
      <td>0.994102</td>
      <td>-0.914158</td>
      <td>-0.391843</td>
      <td>0.755424</td>
      <td>-0.626352</td>
      <td>-0.699599</td>
      <td>0.812026</td>
      <td>-0.691496</td>
      <td>-0.645755</td>
      <td>0.763632</td>
      <td>-0.844808</td>
      <td>-0.985306</td>
      <td>1.000000</td>
      <td>-0.914393</td>
      <td>-0.184860</td>
      <td>0.107382</td>
      <td>-0.690724</td>
      <td>0.826157</td>
      <td>-0.678634</td>
    </tr>
    <tr>
      <th>redshift_upper</th>
      <td>0.673544</td>
      <td>-0.866802</td>
      <td>0.754730</td>
      <td>0.463283</td>
      <td>-0.610215</td>
      <td>0.830086</td>
      <td>-0.484218</td>
      <td>-0.638689</td>
      <td>0.691294</td>
      <td>0.886440</td>
      <td>-0.907620</td>
      <td>0.995767</td>
      <td>0.409495</td>
      <td>-0.778233</td>
      <td>0.684676</td>
      <td>0.644463</td>
      <td>-0.876042</td>
      <td>0.782872</td>
      <td>0.547030</td>
      <td>-0.716411</td>
      <td>0.865053</td>
      <td>0.886252</td>
      <td>-0.914393</td>
      <td>1.000000</td>
      <td>0.459321</td>
      <td>-0.231027</td>
      <td>0.619714</td>
      <td>-0.889823</td>
      <td>0.775586</td>
    </tr>
    <tr>
      <th>far</th>
      <td>-0.007988</td>
      <td>-0.213490</td>
      <td>0.284003</td>
      <td>-0.086445</td>
      <td>-0.019749</td>
      <td>0.177403</td>
      <td>-0.224901</td>
      <td>-0.357351</td>
      <td>0.601730</td>
      <td>0.138640</td>
      <td>-0.175960</td>
      <td>0.458951</td>
      <td>0.118732</td>
      <td>-0.287039</td>
      <td>0.266861</td>
      <td>0.013607</td>
      <td>-0.312846</td>
      <td>0.212604</td>
      <td>-0.076844</td>
      <td>-0.023585</td>
      <td>0.250440</td>
      <td>0.139046</td>
      <td>-0.184860</td>
      <td>0.459321</td>
      <td>1.000000</td>
      <td>-0.317343</td>
      <td>-0.018819</td>
      <td>-0.314554</td>
      <td>0.222843</td>
    </tr>
    <tr>
      <th>p_astro</th>
      <td>0.093121</td>
      <td>0.153595</td>
      <td>-0.176180</td>
      <td>0.222898</td>
      <td>-0.049260</td>
      <td>-0.075916</td>
      <td>0.387586</td>
      <td>0.150394</td>
      <td>-0.182484</td>
      <td>-0.100031</td>
      <td>0.141725</td>
      <td>-0.254961</td>
      <td>-0.078059</td>
      <td>0.008903</td>
      <td>0.064235</td>
      <td>0.051566</td>
      <td>0.193786</td>
      <td>-0.191058</td>
      <td>0.188736</td>
      <td>0.046610</td>
      <td>-0.132329</td>
      <td>-0.062069</td>
      <td>0.107382</td>
      <td>-0.231027</td>
      <td>-0.317343</td>
      <td>1.000000</td>
      <td>0.142559</td>
      <td>0.176720</td>
      <td>-0.174811</td>
    </tr>
    <tr>
      <th>final_mass_source</th>
      <td>0.983488</td>
      <td>-0.758022</td>
      <td>0.505128</td>
      <td>0.955395</td>
      <td>-0.930379</td>
      <td>0.895344</td>
      <td>-0.064574</td>
      <td>-0.175326</td>
      <td>0.186712</td>
      <td>0.690123</td>
      <td>-0.670720</td>
      <td>0.604935</td>
      <td>0.125551</td>
      <td>-0.650182</td>
      <td>0.591684</td>
      <td>0.886376</td>
      <td>-0.779342</td>
      <td>0.587551</td>
      <td>0.988833</td>
      <td>-0.932274</td>
      <td>0.865212</td>
      <td>0.711503</td>
      <td>-0.690724</td>
      <td>0.619714</td>
      <td>-0.018819</td>
      <td>0.142559</td>
      <td>1.000000</td>
      <td>-0.815275</td>
      <td>0.576832</td>
    </tr>
    <tr>
      <th>final_mass_source_lower</th>
      <td>-0.849534</td>
      <td>0.937873</td>
      <td>-0.804109</td>
      <td>-0.683857</td>
      <td>0.805391</td>
      <td>-0.919999</td>
      <td>0.352203</td>
      <td>0.514041</td>
      <td>-0.582241</td>
      <td>-0.808712</td>
      <td>0.817090</td>
      <td>-0.884885</td>
      <td>-0.318183</td>
      <td>0.772704</td>
      <td>-0.727054</td>
      <td>-0.792724</td>
      <td>0.974899</td>
      <td>-0.843652</td>
      <td>-0.754979</td>
      <td>0.886512</td>
      <td>-0.946562</td>
      <td>-0.812215</td>
      <td>0.826157</td>
      <td>-0.889823</td>
      <td>-0.314554</td>
      <td>0.176720</td>
      <td>-0.815275</td>
      <td>1.000000</td>
      <td>-0.837544</td>
    </tr>
    <tr>
      <th>final_mass_source_upper</th>
      <td>0.630959</td>
      <td>-0.854995</td>
      <td>0.977386</td>
      <td>0.437540</td>
      <td>-0.589172</td>
      <td>0.696836</td>
      <td>-0.340329</td>
      <td>-0.527470</td>
      <td>0.662470</td>
      <td>0.650764</td>
      <td>-0.662247</td>
      <td>0.767169</td>
      <td>0.318159</td>
      <td>-0.658188</td>
      <td>0.710832</td>
      <td>0.569174</td>
      <td>-0.810728</td>
      <td>0.983461</td>
      <td>0.512854</td>
      <td>-0.674015</td>
      <td>0.782916</td>
      <td>0.655919</td>
      <td>-0.678634</td>
      <td>0.775586</td>
      <td>0.222843</td>
      <td>-0.174811</td>
      <td>0.576832</td>
      <td>-0.837544</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



From this initial correlation matrix, a vast number of the variables had correlation coefficeints, $\rho$ where |$\rho$| was close to 1. Meaning that most of the variables were strongly correlated to each other. This makes intuitive sense as they are all parameters of the same event, however some don't make as much sense being as correlated as they are. To better understand which correlations were most likely true correlations, the next step was to feed the correlation matrix into an algorithm to produce a pearsonr test matrix.


```python

cols = list(cor.columns)
pearsonDict = {}
for i in cols:
    temp = {}
    for j in cols:
        r, temp[j] = pearsonr(cor[i], cor[j])
    pearsonDict[i] = temp        

pearsonDf = pd.DataFrame.from_dict(pearsonDict)

pearsonDf
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mass_1_source</th>
      <th>mass_1_source_lower</th>
      <th>mass_1_source_upper</th>
      <th>mass_2_source</th>
      <th>mass_2_source_lower</th>
      <th>mass_2_source_upper</th>
      <th>network_matched_filter_snr</th>
      <th>network_matched_filter_snr_lower</th>
      <th>network_matched_filter_snr_upper</th>
      <th>luminosity_distance</th>
      <th>luminosity_distance_lower</th>
      <th>luminosity_distance_upper</th>
      <th>chi_eff</th>
      <th>chi_eff_lower</th>
      <th>chi_eff_upper</th>
      <th>total_mass_source</th>
      <th>total_mass_source_lower</th>
      <th>total_mass_source_upper</th>
      <th>chirp_mass_source</th>
      <th>chirp_mass_source_lower</th>
      <th>chirp_mass_source_upper</th>
      <th>redshift</th>
      <th>redshift_lower</th>
      <th>redshift_upper</th>
      <th>far</th>
      <th>p_astro</th>
      <th>final_mass_source</th>
      <th>final_mass_source_lower</th>
      <th>final_mass_source_upper</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mass_1_source</th>
      <td>0.000000e+00</td>
      <td>2.952065e-17</td>
      <td>1.246870e-11</td>
      <td>7.681062e-21</td>
      <td>7.144420e-31</td>
      <td>1.413249e-25</td>
      <td>2.186359e-05</td>
      <td>2.249948e-06</td>
      <td>3.867973e-06</td>
      <td>3.136846e-16</td>
      <td>9.057529e-16</td>
      <td>4.357122e-13</td>
      <td>4.601379e-05</td>
      <td>1.406911e-15</td>
      <td>1.812327e-14</td>
      <td>7.722088e-33</td>
      <td>3.368283e-18</td>
      <td>1.532938e-13</td>
      <td>5.192259e-26</td>
      <td>1.630534e-33</td>
      <td>2.890195e-22</td>
      <td>1.037333e-16</td>
      <td>3.820235e-16</td>
      <td>2.758241e-13</td>
      <td>1.549977e-02</td>
      <td>0.229203</td>
      <td>3.508650e-34</td>
      <td>1.231781e-18</td>
      <td>3.806204e-13</td>
    </tr>
    <tr>
      <th>mass_1_source_lower</th>
      <td>2.952065e-17</td>
      <td>7.230709e-213</td>
      <td>8.094217e-21</td>
      <td>3.753804e-11</td>
      <td>2.104759e-15</td>
      <td>3.659050e-23</td>
      <td>1.609188e-08</td>
      <td>1.434990e-10</td>
      <td>2.236819e-10</td>
      <td>1.007265e-22</td>
      <td>4.911785e-23</td>
      <td>8.289530e-24</td>
      <td>2.383769e-07</td>
      <td>6.253126e-23</td>
      <td>4.392393e-20</td>
      <td>3.753790e-17</td>
      <td>5.474945e-32</td>
      <td>7.705030e-25</td>
      <td>5.886118e-13</td>
      <td>1.168080e-18</td>
      <td>6.214730e-26</td>
      <td>6.433552e-23</td>
      <td>1.256152e-23</td>
      <td>2.525131e-24</td>
      <td>3.388679e-04</td>
      <td>0.025754</td>
      <td>5.845433e-15</td>
      <td>6.873241e-34</td>
      <td>5.157248e-24</td>
    </tr>
    <tr>
      <th>mass_1_source_upper</th>
      <td>1.246870e-11</td>
      <td>8.094217e-21</td>
      <td>0.000000e+00</td>
      <td>6.090546e-08</td>
      <td>8.245046e-11</td>
      <td>2.044474e-14</td>
      <td>1.111211e-09</td>
      <td>4.955593e-13</td>
      <td>7.658591e-14</td>
      <td>2.675472e-15</td>
      <td>1.079682e-15</td>
      <td>9.388695e-20</td>
      <td>1.245116e-07</td>
      <td>1.263230e-17</td>
      <td>3.071341e-18</td>
      <td>1.212501e-11</td>
      <td>4.895309e-19</td>
      <td>1.388283e-30</td>
      <td>4.210593e-09</td>
      <td>1.565564e-12</td>
      <td>5.386165e-16</td>
      <td>2.843388e-15</td>
      <td>6.506532e-16</td>
      <td>5.066937e-20</td>
      <td>3.434831e-05</td>
      <td>0.007280</td>
      <td>2.408078e-10</td>
      <td>4.748794e-19</td>
      <td>4.057670e-34</td>
    </tr>
    <tr>
      <th>mass_2_source</th>
      <td>7.681062e-21</td>
      <td>3.753804e-11</td>
      <td>6.090546e-08</td>
      <td>7.230709e-213</td>
      <td>1.989998e-23</td>
      <td>1.815141e-15</td>
      <td>9.768584e-04</td>
      <td>2.682206e-04</td>
      <td>3.905596e-04</td>
      <td>5.153927e-11</td>
      <td>1.051495e-10</td>
      <td>8.028237e-09</td>
      <td>1.670918e-03</td>
      <td>8.876831e-11</td>
      <td>2.147474e-10</td>
      <td>5.230305e-20</td>
      <td>6.811159e-12</td>
      <td>3.427911e-09</td>
      <td>1.031650e-32</td>
      <td>6.039683e-19</td>
      <td>4.198612e-14</td>
      <td>2.524420e-11</td>
      <td>6.254192e-11</td>
      <td>5.890683e-09</td>
      <td>9.998037e-02</td>
      <td>0.629649</td>
      <td>2.892281e-25</td>
      <td>4.349177e-12</td>
      <td>6.051949e-09</td>
    </tr>
    <tr>
      <th>mass_2_source_lower</th>
      <td>7.144420e-31</td>
      <td>2.104759e-15</td>
      <td>8.245046e-11</td>
      <td>1.989998e-23</td>
      <td>7.230709e-213</td>
      <td>1.468906e-22</td>
      <td>4.244728e-05</td>
      <td>7.840119e-06</td>
      <td>1.101391e-05</td>
      <td>1.822501e-14</td>
      <td>3.847795e-14</td>
      <td>6.183032e-12</td>
      <td>1.826611e-04</td>
      <td>2.020617e-14</td>
      <td>6.089007e-14</td>
      <td>2.842232e-29</td>
      <td>1.206024e-16</td>
      <td>1.453581e-12</td>
      <td>7.009607e-29</td>
      <td>8.853396e-31</td>
      <td>2.491595e-20</td>
      <td>6.805960e-15</td>
      <td>1.705361e-14</td>
      <td>3.917765e-12</td>
      <td>2.002270e-02</td>
      <td>0.251361</td>
      <td>2.699745e-34</td>
      <td>5.957291e-17</td>
      <td>3.324090e-12</td>
    </tr>
    <tr>
      <th>mass_2_source_upper</th>
      <td>1.413249e-25</td>
      <td>3.659050e-23</td>
      <td>2.044474e-14</td>
      <td>1.815141e-15</td>
      <td>1.468906e-22</td>
      <td>7.230709e-213</td>
      <td>7.123515e-07</td>
      <td>5.496295e-08</td>
      <td>9.464447e-08</td>
      <td>4.631302e-21</td>
      <td>9.894113e-21</td>
      <td>1.889477e-17</td>
      <td>6.205122e-06</td>
      <td>7.489701e-20</td>
      <td>1.312369e-17</td>
      <td>2.437694e-25</td>
      <td>2.268857e-25</td>
      <td>6.163557e-17</td>
      <td>3.022607e-18</td>
      <td>1.367952e-29</td>
      <td>3.959121e-36</td>
      <td>1.053057e-21</td>
      <td>2.388793e-21</td>
      <td>9.751849e-18</td>
      <td>2.338337e-03</td>
      <td>0.078791</td>
      <td>1.343855e-21</td>
      <td>2.689311e-26</td>
      <td>2.233381e-16</td>
    </tr>
    <tr>
      <th>network_matched_filter_snr</th>
      <td>2.186359e-05</td>
      <td>1.609188e-08</td>
      <td>1.111211e-09</td>
      <td>9.768584e-04</td>
      <td>4.244728e-05</td>
      <td>7.123515e-07</td>
      <td>7.230709e-213</td>
      <td>1.293376e-14</td>
      <td>2.120415e-12</td>
      <td>6.127532e-09</td>
      <td>3.612820e-09</td>
      <td>1.341754e-10</td>
      <td>1.249804e-06</td>
      <td>2.737687e-09</td>
      <td>4.615810e-09</td>
      <td>1.569187e-05</td>
      <td>3.001843e-08</td>
      <td>3.837445e-09</td>
      <td>2.758899e-04</td>
      <td>6.699207e-06</td>
      <td>2.374753e-07</td>
      <td>7.580384e-09</td>
      <td>3.771030e-09</td>
      <td>1.348714e-10</td>
      <td>1.020624e-06</td>
      <td>0.000120</td>
      <td>7.768916e-05</td>
      <td>3.594926e-08</td>
      <td>3.101063e-09</td>
    </tr>
    <tr>
      <th>network_matched_filter_snr_lower</th>
      <td>2.249948e-06</td>
      <td>1.434990e-10</td>
      <td>4.955593e-13</td>
      <td>2.682206e-04</td>
      <td>7.840119e-06</td>
      <td>5.496295e-08</td>
      <td>1.293376e-14</td>
      <td>0.000000e+00</td>
      <td>3.142143e-23</td>
      <td>6.673105e-10</td>
      <td>3.838629e-10</td>
      <td>8.258444e-13</td>
      <td>3.468067e-07</td>
      <td>1.964702e-11</td>
      <td>2.688929e-11</td>
      <td>1.439036e-06</td>
      <td>3.795220e-10</td>
      <td>5.489724e-12</td>
      <td>5.848340e-05</td>
      <td>9.163638e-07</td>
      <td>1.254976e-08</td>
      <td>7.625626e-10</td>
      <td>3.406929e-10</td>
      <td>6.956260e-13</td>
      <td>1.090325e-07</td>
      <td>0.001586</td>
      <td>1.138962e-05</td>
      <td>5.878885e-10</td>
      <td>4.883177e-12</td>
    </tr>
    <tr>
      <th>network_matched_filter_snr_upper</th>
      <td>3.867973e-06</td>
      <td>2.236819e-10</td>
      <td>7.658591e-14</td>
      <td>3.905596e-04</td>
      <td>1.101391e-05</td>
      <td>9.464447e-08</td>
      <td>2.120415e-12</td>
      <td>3.142143e-23</td>
      <td>0.000000e+00</td>
      <td>4.185701e-09</td>
      <td>2.189670e-09</td>
      <td>1.932310e-12</td>
      <td>5.600883e-07</td>
      <td>6.924753e-11</td>
      <td>5.057236e-11</td>
      <td>2.690050e-06</td>
      <td>4.824583e-10</td>
      <td>2.866801e-12</td>
      <td>9.132304e-05</td>
      <td>1.460744e-06</td>
      <td>1.750544e-08</td>
      <td>4.807895e-09</td>
      <td>1.917457e-09</td>
      <td>1.711595e-12</td>
      <td>2.387652e-09</td>
      <td>0.000676</td>
      <td>1.816268e-05</td>
      <td>7.041197e-10</td>
      <td>1.951242e-12</td>
    </tr>
    <tr>
      <th>luminosity_distance</th>
      <td>3.136846e-16</td>
      <td>1.007265e-22</td>
      <td>2.675472e-15</td>
      <td>5.153927e-11</td>
      <td>1.822501e-14</td>
      <td>4.631302e-21</td>
      <td>6.127532e-09</td>
      <td>6.673105e-10</td>
      <td>4.185701e-09</td>
      <td>7.230709e-213</td>
      <td>5.236676e-46</td>
      <td>4.644406e-23</td>
      <td>1.048579e-07</td>
      <td>1.513719e-20</td>
      <td>1.915785e-16</td>
      <td>2.615198e-16</td>
      <td>7.960237e-22</td>
      <td>2.830820e-17</td>
      <td>1.095267e-12</td>
      <td>2.752652e-17</td>
      <td>1.222944e-21</td>
      <td>1.270112e-47</td>
      <td>1.966139e-44</td>
      <td>4.372988e-23</td>
      <td>7.437117e-04</td>
      <td>0.033895</td>
      <td>2.275263e-14</td>
      <td>4.024662e-22</td>
      <td>7.227964e-17</td>
    </tr>
    <tr>
      <th>luminosity_distance_lower</th>
      <td>9.057529e-16</td>
      <td>4.911785e-23</td>
      <td>1.079682e-15</td>
      <td>1.051495e-10</td>
      <td>3.847795e-14</td>
      <td>9.894113e-21</td>
      <td>3.612820e-09</td>
      <td>3.838629e-10</td>
      <td>2.189670e-09</td>
      <td>5.236676e-46</td>
      <td>0.000000e+00</td>
      <td>1.874781e-24</td>
      <td>8.675852e-08</td>
      <td>1.087397e-20</td>
      <td>1.432012e-16</td>
      <td>7.143443e-16</td>
      <td>4.009860e-22</td>
      <td>1.171498e-17</td>
      <td>2.526618e-12</td>
      <td>6.562725e-17</td>
      <td>1.391319e-21</td>
      <td>2.821923e-40</td>
      <td>2.386156e-47</td>
      <td>2.080660e-24</td>
      <td>5.040167e-04</td>
      <td>0.026469</td>
      <td>5.739631e-14</td>
      <td>2.157047e-22</td>
      <td>2.960730e-17</td>
    </tr>
    <tr>
      <th>luminosity_distance_upper</th>
      <td>4.357122e-13</td>
      <td>8.289530e-24</td>
      <td>9.388695e-20</td>
      <td>8.028237e-09</td>
      <td>6.183032e-12</td>
      <td>1.889477e-17</td>
      <td>1.341754e-10</td>
      <td>8.258444e-13</td>
      <td>1.932310e-12</td>
      <td>4.644406e-23</td>
      <td>1.874781e-24</td>
      <td>7.230709e-213</td>
      <td>3.360957e-08</td>
      <td>1.850443e-21</td>
      <td>7.814957e-18</td>
      <td>2.861636e-13</td>
      <td>1.197804e-22</td>
      <td>4.721540e-21</td>
      <td>3.772910e-10</td>
      <td>3.738458e-14</td>
      <td>3.728574e-19</td>
      <td>1.337448e-22</td>
      <td>1.914095e-24</td>
      <td>1.264076e-49</td>
      <td>2.208681e-05</td>
      <td>0.007037</td>
      <td>1.381150e-11</td>
      <td>1.595079e-22</td>
      <td>1.011244e-20</td>
    </tr>
    <tr>
      <th>chi_eff</th>
      <td>4.601379e-05</td>
      <td>2.383769e-07</td>
      <td>1.245116e-07</td>
      <td>1.670918e-03</td>
      <td>1.826611e-04</td>
      <td>6.205122e-06</td>
      <td>1.249804e-06</td>
      <td>3.468067e-07</td>
      <td>5.600883e-07</td>
      <td>1.048579e-07</td>
      <td>8.675852e-08</td>
      <td>3.360957e-08</td>
      <td>0.000000e+00</td>
      <td>3.811511e-06</td>
      <td>1.623952e-05</td>
      <td>6.785248e-05</td>
      <td>1.265358e-06</td>
      <td>4.434390e-07</td>
      <td>4.835460e-04</td>
      <td>3.501432e-05</td>
      <td>4.506714e-06</td>
      <td>1.754188e-07</td>
      <td>1.360661e-07</td>
      <td>5.136965e-08</td>
      <td>5.514387e-04</td>
      <td>0.005168</td>
      <td>1.643364e-04</td>
      <td>1.045957e-06</td>
      <td>3.064566e-07</td>
    </tr>
    <tr>
      <th>chi_eff_lower</th>
      <td>1.406911e-15</td>
      <td>6.253126e-23</td>
      <td>1.263230e-17</td>
      <td>8.876831e-11</td>
      <td>2.020617e-14</td>
      <td>7.489701e-20</td>
      <td>2.737687e-09</td>
      <td>1.964702e-11</td>
      <td>6.924753e-11</td>
      <td>1.513719e-20</td>
      <td>1.087397e-20</td>
      <td>1.850443e-21</td>
      <td>3.811511e-06</td>
      <td>0.000000e+00</td>
      <td>3.503246e-26</td>
      <td>8.037038e-16</td>
      <td>2.330415e-23</td>
      <td>9.521977e-20</td>
      <td>2.656778e-12</td>
      <td>1.304915e-16</td>
      <td>2.153172e-21</td>
      <td>2.899202e-21</td>
      <td>1.015888e-21</td>
      <td>2.043372e-22</td>
      <td>1.666088e-04</td>
      <td>0.044161</td>
      <td>5.591434e-14</td>
      <td>1.699320e-23</td>
      <td>2.652554e-19</td>
    </tr>
    <tr>
      <th>chi_eff_upper</th>
      <td>1.812327e-14</td>
      <td>4.392393e-20</td>
      <td>3.071341e-18</td>
      <td>2.147474e-10</td>
      <td>6.089007e-14</td>
      <td>1.312369e-17</td>
      <td>4.615810e-09</td>
      <td>2.688929e-11</td>
      <td>5.057236e-11</td>
      <td>1.915785e-16</td>
      <td>1.432012e-16</td>
      <td>7.814957e-18</td>
      <td>1.623952e-05</td>
      <td>3.503246e-26</td>
      <td>7.230709e-213</td>
      <td>9.867745e-15</td>
      <td>1.256422e-20</td>
      <td>5.048458e-20</td>
      <td>9.730297e-12</td>
      <td>1.440978e-15</td>
      <td>5.089606e-19</td>
      <td>5.932545e-17</td>
      <td>2.696547e-17</td>
      <td>1.453767e-18</td>
      <td>1.903353e-04</td>
      <td>0.053881</td>
      <td>3.429017e-13</td>
      <td>1.171875e-20</td>
      <td>9.215536e-20</td>
    </tr>
    <tr>
      <th>total_mass_source</th>
      <td>7.722088e-33</td>
      <td>3.753790e-17</td>
      <td>1.212501e-11</td>
      <td>5.230305e-20</td>
      <td>2.842232e-29</td>
      <td>2.437694e-25</td>
      <td>1.569187e-05</td>
      <td>1.439036e-06</td>
      <td>2.690050e-06</td>
      <td>2.615198e-16</td>
      <td>7.143443e-16</td>
      <td>2.861636e-13</td>
      <td>6.785248e-05</td>
      <td>8.037038e-16</td>
      <td>9.867745e-15</td>
      <td>0.000000e+00</td>
      <td>8.718207e-19</td>
      <td>9.776448e-14</td>
      <td>3.810319e-24</td>
      <td>5.460683e-32</td>
      <td>1.602768e-22</td>
      <td>8.232555e-17</td>
      <td>2.861371e-16</td>
      <td>1.730558e-13</td>
      <td>1.260608e-02</td>
      <td>0.202778</td>
      <td>2.296272e-29</td>
      <td>8.022102e-19</td>
      <td>3.429746e-13</td>
    </tr>
    <tr>
      <th>total_mass_source_lower</th>
      <td>3.368283e-18</td>
      <td>5.474945e-32</td>
      <td>4.895309e-19</td>
      <td>6.811159e-12</td>
      <td>1.206024e-16</td>
      <td>2.268857e-25</td>
      <td>3.001843e-08</td>
      <td>3.795220e-10</td>
      <td>4.824583e-10</td>
      <td>7.960237e-22</td>
      <td>4.009860e-22</td>
      <td>1.197804e-22</td>
      <td>1.265358e-06</td>
      <td>2.330415e-23</td>
      <td>1.256422e-20</td>
      <td>8.718207e-19</td>
      <td>0.000000e+00</td>
      <td>4.183691e-23</td>
      <td>9.380646e-14</td>
      <td>3.628998e-20</td>
      <td>4.903926e-30</td>
      <td>3.303447e-22</td>
      <td>7.841804e-23</td>
      <td>2.964920e-23</td>
      <td>2.902308e-04</td>
      <td>0.026712</td>
      <td>6.615027e-16</td>
      <td>7.907930e-43</td>
      <td>7.386841e-22</td>
    </tr>
    <tr>
      <th>total_mass_source_upper</th>
      <td>1.532938e-13</td>
      <td>7.705030e-25</td>
      <td>1.388283e-30</td>
      <td>3.427911e-09</td>
      <td>1.453581e-12</td>
      <td>6.163557e-17</td>
      <td>3.837445e-09</td>
      <td>5.489724e-12</td>
      <td>2.866801e-12</td>
      <td>2.830820e-17</td>
      <td>1.171498e-17</td>
      <td>4.721540e-21</td>
      <td>4.434390e-07</td>
      <td>9.521977e-20</td>
      <td>5.048458e-20</td>
      <td>9.776448e-14</td>
      <td>4.183691e-23</td>
      <td>0.000000e+00</td>
      <td>1.526926e-10</td>
      <td>1.052778e-14</td>
      <td>5.305639e-19</td>
      <td>2.508566e-17</td>
      <td>5.416365e-18</td>
      <td>1.783134e-21</td>
      <td>1.157030e-04</td>
      <td>0.012132</td>
      <td>5.231721e-12</td>
      <td>8.250429e-23</td>
      <td>4.260012e-42</td>
    </tr>
    <tr>
      <th>chirp_mass_source</th>
      <td>5.192259e-26</td>
      <td>5.886118e-13</td>
      <td>4.210593e-09</td>
      <td>1.031650e-32</td>
      <td>7.009607e-29</td>
      <td>3.022607e-18</td>
      <td>2.758899e-04</td>
      <td>5.848340e-05</td>
      <td>9.132304e-05</td>
      <td>1.095267e-12</td>
      <td>2.526618e-12</td>
      <td>3.772910e-10</td>
      <td>4.835460e-04</td>
      <td>2.656778e-12</td>
      <td>9.730297e-12</td>
      <td>3.810319e-24</td>
      <td>9.380646e-14</td>
      <td>1.526926e-10</td>
      <td>8.376948e-209</td>
      <td>6.981041e-23</td>
      <td>1.888034e-16</td>
      <td>4.763282e-13</td>
      <td>1.370488e-12</td>
      <td>2.657937e-10</td>
      <td>5.650763e-02</td>
      <td>0.462557</td>
      <td>8.649392e-34</td>
      <td>5.217599e-14</td>
      <td>2.945969e-10</td>
    </tr>
    <tr>
      <th>chirp_mass_source_lower</th>
      <td>1.630534e-33</td>
      <td>1.168080e-18</td>
      <td>1.565564e-12</td>
      <td>6.039683e-19</td>
      <td>8.853396e-31</td>
      <td>1.367952e-29</td>
      <td>6.699207e-06</td>
      <td>9.163638e-07</td>
      <td>1.460744e-06</td>
      <td>2.752652e-17</td>
      <td>6.562725e-17</td>
      <td>3.738458e-14</td>
      <td>3.501432e-05</td>
      <td>1.304915e-16</td>
      <td>1.440978e-15</td>
      <td>5.460683e-32</td>
      <td>3.628998e-20</td>
      <td>1.052778e-14</td>
      <td>6.981041e-23</td>
      <td>7.230709e-213</td>
      <td>1.164092e-25</td>
      <td>8.693180e-18</td>
      <td>2.471273e-17</td>
      <td>2.248853e-14</td>
      <td>9.049294e-03</td>
      <td>0.142700</td>
      <td>6.134334e-28</td>
      <td>1.301647e-20</td>
      <td>3.070662e-14</td>
    </tr>
    <tr>
      <th>chirp_mass_source_upper</th>
      <td>2.890195e-22</td>
      <td>6.214730e-26</td>
      <td>5.386165e-16</td>
      <td>4.198612e-14</td>
      <td>2.491595e-20</td>
      <td>3.959121e-36</td>
      <td>2.374753e-07</td>
      <td>1.254976e-08</td>
      <td>1.750544e-08</td>
      <td>1.222944e-21</td>
      <td>1.391319e-21</td>
      <td>3.728574e-19</td>
      <td>4.506714e-06</td>
      <td>2.153172e-21</td>
      <td>5.089606e-19</td>
      <td>1.602768e-22</td>
      <td>4.903926e-30</td>
      <td>5.305639e-19</td>
      <td>1.888034e-16</td>
      <td>1.164092e-25</td>
      <td>0.000000e+00</td>
      <td>3.086507e-22</td>
      <td>2.654492e-22</td>
      <td>1.596989e-19</td>
      <td>1.097560e-03</td>
      <td>0.051425</td>
      <td>3.133565e-19</td>
      <td>2.095692e-31</td>
      <td>2.631119e-18</td>
    </tr>
    <tr>
      <th>redshift</th>
      <td>1.037333e-16</td>
      <td>6.433552e-23</td>
      <td>2.843388e-15</td>
      <td>2.524420e-11</td>
      <td>6.805960e-15</td>
      <td>1.053057e-21</td>
      <td>7.580384e-09</td>
      <td>7.625626e-10</td>
      <td>4.807895e-09</td>
      <td>1.270112e-47</td>
      <td>2.821923e-40</td>
      <td>1.337448e-22</td>
      <td>1.754188e-07</td>
      <td>2.899202e-21</td>
      <td>5.932545e-17</td>
      <td>8.232555e-17</td>
      <td>3.303447e-22</td>
      <td>2.508566e-17</td>
      <td>4.763282e-13</td>
      <td>8.693180e-18</td>
      <td>3.086507e-22</td>
      <td>0.000000e+00</td>
      <td>1.201425e-43</td>
      <td>9.316592e-23</td>
      <td>8.366689e-04</td>
      <td>0.040483</td>
      <td>8.579264e-15</td>
      <td>1.566297e-22</td>
      <td>6.696124e-17</td>
    </tr>
    <tr>
      <th>redshift_lower</th>
      <td>3.820235e-16</td>
      <td>1.256152e-23</td>
      <td>6.506532e-16</td>
      <td>6.254192e-11</td>
      <td>1.705361e-14</td>
      <td>2.388793e-21</td>
      <td>3.771030e-09</td>
      <td>3.406929e-10</td>
      <td>1.917457e-09</td>
      <td>1.966139e-44</td>
      <td>2.386156e-47</td>
      <td>1.914095e-24</td>
      <td>1.360661e-07</td>
      <td>1.015888e-21</td>
      <td>2.696547e-17</td>
      <td>2.861371e-16</td>
      <td>7.841804e-23</td>
      <td>5.416365e-18</td>
      <td>1.370488e-12</td>
      <td>2.471273e-17</td>
      <td>2.654492e-22</td>
      <td>1.201425e-43</td>
      <td>0.000000e+00</td>
      <td>1.375399e-24</td>
      <td>5.135346e-04</td>
      <td>0.030439</td>
      <td>2.729509e-14</td>
      <td>3.870670e-23</td>
      <td>1.444684e-17</td>
    </tr>
    <tr>
      <th>redshift_upper</th>
      <td>2.758241e-13</td>
      <td>2.525131e-24</td>
      <td>5.066937e-20</td>
      <td>5.890683e-09</td>
      <td>3.917765e-12</td>
      <td>9.751849e-18</td>
      <td>1.348714e-10</td>
      <td>6.956260e-13</td>
      <td>1.711595e-12</td>
      <td>4.372988e-23</td>
      <td>2.080660e-24</td>
      <td>1.264076e-49</td>
      <td>5.136965e-08</td>
      <td>2.043372e-22</td>
      <td>1.453767e-18</td>
      <td>1.730558e-13</td>
      <td>2.964920e-23</td>
      <td>1.783134e-21</td>
      <td>2.657937e-10</td>
      <td>2.248853e-14</td>
      <td>1.596989e-19</td>
      <td>9.316592e-23</td>
      <td>1.375399e-24</td>
      <td>7.230709e-213</td>
      <td>2.361499e-05</td>
      <td>0.008143</td>
      <td>9.162416e-12</td>
      <td>4.200791e-23</td>
      <td>4.223580e-21</td>
    </tr>
    <tr>
      <th>far</th>
      <td>1.549977e-02</td>
      <td>3.388679e-04</td>
      <td>3.434831e-05</td>
      <td>9.998037e-02</td>
      <td>2.002270e-02</td>
      <td>2.338337e-03</td>
      <td>1.020624e-06</td>
      <td>1.090325e-07</td>
      <td>2.387652e-09</td>
      <td>7.437117e-04</td>
      <td>5.040167e-04</td>
      <td>2.208681e-05</td>
      <td>5.514387e-04</td>
      <td>1.666088e-04</td>
      <td>1.903353e-04</td>
      <td>1.260608e-02</td>
      <td>2.902308e-04</td>
      <td>1.157030e-04</td>
      <td>5.650763e-02</td>
      <td>9.049294e-03</td>
      <td>1.097560e-03</td>
      <td>8.366689e-04</td>
      <td>5.135346e-04</td>
      <td>2.361499e-05</td>
      <td>0.000000e+00</td>
      <td>0.000010</td>
      <td>2.788954e-02</td>
      <td>3.348846e-04</td>
      <td>1.010219e-04</td>
    </tr>
    <tr>
      <th>p_astro</th>
      <td>2.292026e-01</td>
      <td>2.575412e-02</td>
      <td>7.280247e-03</td>
      <td>6.296488e-01</td>
      <td>2.513610e-01</td>
      <td>7.879110e-02</td>
      <td>1.199252e-04</td>
      <td>1.586420e-03</td>
      <td>6.761854e-04</td>
      <td>3.389481e-02</td>
      <td>2.646942e-02</td>
      <td>7.037256e-03</td>
      <td>5.167684e-03</td>
      <td>4.416088e-02</td>
      <td>5.388112e-02</td>
      <td>2.027781e-01</td>
      <td>2.671188e-02</td>
      <td>1.213166e-02</td>
      <td>4.625574e-01</td>
      <td>1.426996e-01</td>
      <td>5.142546e-02</td>
      <td>4.048327e-02</td>
      <td>3.043913e-02</td>
      <td>8.143217e-03</td>
      <td>9.710429e-06</td>
      <td>0.000000</td>
      <td>3.262720e-01</td>
      <td>2.934673e-02</td>
      <td>1.151923e-02</td>
    </tr>
    <tr>
      <th>final_mass_source</th>
      <td>3.508650e-34</td>
      <td>5.845433e-15</td>
      <td>2.408078e-10</td>
      <td>2.892281e-25</td>
      <td>2.699745e-34</td>
      <td>1.343855e-21</td>
      <td>7.768916e-05</td>
      <td>1.138962e-05</td>
      <td>1.816268e-05</td>
      <td>2.275263e-14</td>
      <td>5.739631e-14</td>
      <td>1.381150e-11</td>
      <td>1.643364e-04</td>
      <td>5.591434e-14</td>
      <td>3.429017e-13</td>
      <td>2.296272e-29</td>
      <td>6.615027e-16</td>
      <td>5.231721e-12</td>
      <td>8.649392e-34</td>
      <td>6.134334e-28</td>
      <td>3.133565e-19</td>
      <td>8.579264e-15</td>
      <td>2.729509e-14</td>
      <td>9.162416e-12</td>
      <td>2.788954e-02</td>
      <td>0.326272</td>
      <td>0.000000e+00</td>
      <td>3.144979e-16</td>
      <td>1.142456e-11</td>
    </tr>
    <tr>
      <th>final_mass_source_lower</th>
      <td>1.231781e-18</td>
      <td>6.873241e-34</td>
      <td>4.748794e-19</td>
      <td>4.349177e-12</td>
      <td>5.957291e-17</td>
      <td>2.689311e-26</td>
      <td>3.594926e-08</td>
      <td>5.878885e-10</td>
      <td>7.041197e-10</td>
      <td>4.024662e-22</td>
      <td>2.157047e-22</td>
      <td>1.595079e-22</td>
      <td>1.045957e-06</td>
      <td>1.699320e-23</td>
      <td>1.171875e-20</td>
      <td>8.022102e-19</td>
      <td>7.907930e-43</td>
      <td>8.250429e-23</td>
      <td>5.217599e-14</td>
      <td>1.301647e-20</td>
      <td>2.095692e-31</td>
      <td>1.566297e-22</td>
      <td>3.870670e-23</td>
      <td>4.200791e-23</td>
      <td>3.348846e-04</td>
      <td>0.029347</td>
      <td>3.144979e-16</td>
      <td>7.230709e-213</td>
      <td>7.024697e-22</td>
    </tr>
    <tr>
      <th>final_mass_source_upper</th>
      <td>3.806204e-13</td>
      <td>5.157248e-24</td>
      <td>4.057670e-34</td>
      <td>6.051949e-09</td>
      <td>3.324090e-12</td>
      <td>2.233381e-16</td>
      <td>3.101063e-09</td>
      <td>4.883177e-12</td>
      <td>1.951242e-12</td>
      <td>7.227964e-17</td>
      <td>2.960730e-17</td>
      <td>1.011244e-20</td>
      <td>3.064566e-07</td>
      <td>2.652554e-19</td>
      <td>9.215536e-20</td>
      <td>3.429746e-13</td>
      <td>7.386841e-22</td>
      <td>4.260012e-42</td>
      <td>2.945969e-10</td>
      <td>3.070662e-14</td>
      <td>2.631119e-18</td>
      <td>6.696124e-17</td>
      <td>1.444684e-17</td>
      <td>4.223580e-21</td>
      <td>1.010219e-04</td>
      <td>0.011519</td>
      <td>1.142456e-11</td>
      <td>7.024697e-22</td>
      <td>0.000000e+00</td>
    </tr>
  </tbody>
</table>
</div>



Soem obvious results from this are that there is a high probability that the correlation coefficients of upper and lower bounds in reference to the variables they are bounded on are highly unlikely to be found in uncorrelated data. This is a pretty obvious and expected result. To find more nuanced and hidden correlations tje maxtric was narrowed down to only correlations with a pearsonr probability of less than 0.05.  


```python
def onlySignificant(pearson, thresh):
    col = pearson.columns
    goodPearson = {}
    for i in col:
        temp = {}
        for j in col:
            if pearson[i][j] <= thresh:
                temp[j] = pearson[i][j]
        goodPearson[i] = temp
    return goodPearson
gP = onlySignificant(pearsonDf, 0.05)

gPdf = pd.DataFrame.from_dict(gP)
gPdf
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mass_1_source</th>
      <th>mass_1_source_lower</th>
      <th>mass_1_source_upper</th>
      <th>mass_2_source</th>
      <th>mass_2_source_lower</th>
      <th>mass_2_source_upper</th>
      <th>network_matched_filter_snr</th>
      <th>network_matched_filter_snr_lower</th>
      <th>network_matched_filter_snr_upper</th>
      <th>luminosity_distance</th>
      <th>luminosity_distance_lower</th>
      <th>luminosity_distance_upper</th>
      <th>chi_eff</th>
      <th>chi_eff_lower</th>
      <th>chi_eff_upper</th>
      <th>total_mass_source</th>
      <th>total_mass_source_lower</th>
      <th>total_mass_source_upper</th>
      <th>chirp_mass_source</th>
      <th>chirp_mass_source_lower</th>
      <th>chirp_mass_source_upper</th>
      <th>redshift</th>
      <th>redshift_lower</th>
      <th>redshift_upper</th>
      <th>far</th>
      <th>p_astro</th>
      <th>final_mass_source</th>
      <th>final_mass_source_lower</th>
      <th>final_mass_source_upper</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mass_1_source</th>
      <td>0.000000e+00</td>
      <td>2.952065e-17</td>
      <td>1.246870e-11</td>
      <td>7.681062e-21</td>
      <td>7.144420e-31</td>
      <td>1.413249e-25</td>
      <td>2.186359e-05</td>
      <td>2.249948e-06</td>
      <td>3.867973e-06</td>
      <td>3.136846e-16</td>
      <td>9.057529e-16</td>
      <td>4.357122e-13</td>
      <td>4.601379e-05</td>
      <td>1.406911e-15</td>
      <td>1.812327e-14</td>
      <td>7.722088e-33</td>
      <td>3.368283e-18</td>
      <td>1.532938e-13</td>
      <td>5.192259e-26</td>
      <td>1.630534e-33</td>
      <td>2.890195e-22</td>
      <td>1.037333e-16</td>
      <td>3.820235e-16</td>
      <td>2.758241e-13</td>
      <td>1.549977e-02</td>
      <td>NaN</td>
      <td>3.508650e-34</td>
      <td>1.231781e-18</td>
      <td>3.806204e-13</td>
    </tr>
    <tr>
      <th>mass_1_source_lower</th>
      <td>2.952065e-17</td>
      <td>7.230709e-213</td>
      <td>8.094217e-21</td>
      <td>3.753804e-11</td>
      <td>2.104759e-15</td>
      <td>3.659050e-23</td>
      <td>1.609188e-08</td>
      <td>1.434990e-10</td>
      <td>2.236819e-10</td>
      <td>1.007265e-22</td>
      <td>4.911785e-23</td>
      <td>8.289530e-24</td>
      <td>2.383769e-07</td>
      <td>6.253126e-23</td>
      <td>4.392393e-20</td>
      <td>3.753790e-17</td>
      <td>5.474945e-32</td>
      <td>7.705030e-25</td>
      <td>5.886118e-13</td>
      <td>1.168080e-18</td>
      <td>6.214730e-26</td>
      <td>6.433552e-23</td>
      <td>1.256152e-23</td>
      <td>2.525131e-24</td>
      <td>3.388679e-04</td>
      <td>0.025754</td>
      <td>5.845433e-15</td>
      <td>6.873241e-34</td>
      <td>5.157248e-24</td>
    </tr>
    <tr>
      <th>mass_1_source_upper</th>
      <td>1.246870e-11</td>
      <td>8.094217e-21</td>
      <td>0.000000e+00</td>
      <td>6.090546e-08</td>
      <td>8.245046e-11</td>
      <td>2.044474e-14</td>
      <td>1.111211e-09</td>
      <td>4.955593e-13</td>
      <td>7.658591e-14</td>
      <td>2.675472e-15</td>
      <td>1.079682e-15</td>
      <td>9.388695e-20</td>
      <td>1.245116e-07</td>
      <td>1.263230e-17</td>
      <td>3.071341e-18</td>
      <td>1.212501e-11</td>
      <td>4.895309e-19</td>
      <td>1.388283e-30</td>
      <td>4.210593e-09</td>
      <td>1.565564e-12</td>
      <td>5.386165e-16</td>
      <td>2.843388e-15</td>
      <td>6.506532e-16</td>
      <td>5.066937e-20</td>
      <td>3.434831e-05</td>
      <td>0.007280</td>
      <td>2.408078e-10</td>
      <td>4.748794e-19</td>
      <td>4.057670e-34</td>
    </tr>
    <tr>
      <th>mass_2_source</th>
      <td>7.681062e-21</td>
      <td>3.753804e-11</td>
      <td>6.090546e-08</td>
      <td>7.230709e-213</td>
      <td>1.989998e-23</td>
      <td>1.815141e-15</td>
      <td>9.768584e-04</td>
      <td>2.682206e-04</td>
      <td>3.905596e-04</td>
      <td>5.153927e-11</td>
      <td>1.051495e-10</td>
      <td>8.028237e-09</td>
      <td>1.670918e-03</td>
      <td>8.876831e-11</td>
      <td>2.147474e-10</td>
      <td>5.230305e-20</td>
      <td>6.811159e-12</td>
      <td>3.427911e-09</td>
      <td>1.031650e-32</td>
      <td>6.039683e-19</td>
      <td>4.198612e-14</td>
      <td>2.524420e-11</td>
      <td>6.254192e-11</td>
      <td>5.890683e-09</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.892281e-25</td>
      <td>4.349177e-12</td>
      <td>6.051949e-09</td>
    </tr>
    <tr>
      <th>mass_2_source_lower</th>
      <td>7.144420e-31</td>
      <td>2.104759e-15</td>
      <td>8.245046e-11</td>
      <td>1.989998e-23</td>
      <td>7.230709e-213</td>
      <td>1.468906e-22</td>
      <td>4.244728e-05</td>
      <td>7.840119e-06</td>
      <td>1.101391e-05</td>
      <td>1.822501e-14</td>
      <td>3.847795e-14</td>
      <td>6.183032e-12</td>
      <td>1.826611e-04</td>
      <td>2.020617e-14</td>
      <td>6.089007e-14</td>
      <td>2.842232e-29</td>
      <td>1.206024e-16</td>
      <td>1.453581e-12</td>
      <td>7.009607e-29</td>
      <td>8.853396e-31</td>
      <td>2.491595e-20</td>
      <td>6.805960e-15</td>
      <td>1.705361e-14</td>
      <td>3.917765e-12</td>
      <td>2.002270e-02</td>
      <td>NaN</td>
      <td>2.699745e-34</td>
      <td>5.957291e-17</td>
      <td>3.324090e-12</td>
    </tr>
    <tr>
      <th>mass_2_source_upper</th>
      <td>1.413249e-25</td>
      <td>3.659050e-23</td>
      <td>2.044474e-14</td>
      <td>1.815141e-15</td>
      <td>1.468906e-22</td>
      <td>7.230709e-213</td>
      <td>7.123515e-07</td>
      <td>5.496295e-08</td>
      <td>9.464447e-08</td>
      <td>4.631302e-21</td>
      <td>9.894113e-21</td>
      <td>1.889477e-17</td>
      <td>6.205122e-06</td>
      <td>7.489701e-20</td>
      <td>1.312369e-17</td>
      <td>2.437694e-25</td>
      <td>2.268857e-25</td>
      <td>6.163557e-17</td>
      <td>3.022607e-18</td>
      <td>1.367952e-29</td>
      <td>3.959121e-36</td>
      <td>1.053057e-21</td>
      <td>2.388793e-21</td>
      <td>9.751849e-18</td>
      <td>2.338337e-03</td>
      <td>NaN</td>
      <td>1.343855e-21</td>
      <td>2.689311e-26</td>
      <td>2.233381e-16</td>
    </tr>
    <tr>
      <th>network_matched_filter_snr</th>
      <td>2.186359e-05</td>
      <td>1.609188e-08</td>
      <td>1.111211e-09</td>
      <td>9.768584e-04</td>
      <td>4.244728e-05</td>
      <td>7.123515e-07</td>
      <td>7.230709e-213</td>
      <td>1.293376e-14</td>
      <td>2.120415e-12</td>
      <td>6.127532e-09</td>
      <td>3.612820e-09</td>
      <td>1.341754e-10</td>
      <td>1.249804e-06</td>
      <td>2.737687e-09</td>
      <td>4.615810e-09</td>
      <td>1.569187e-05</td>
      <td>3.001843e-08</td>
      <td>3.837445e-09</td>
      <td>2.758899e-04</td>
      <td>6.699207e-06</td>
      <td>2.374753e-07</td>
      <td>7.580384e-09</td>
      <td>3.771030e-09</td>
      <td>1.348714e-10</td>
      <td>1.020624e-06</td>
      <td>0.000120</td>
      <td>7.768916e-05</td>
      <td>3.594926e-08</td>
      <td>3.101063e-09</td>
    </tr>
    <tr>
      <th>network_matched_filter_snr_lower</th>
      <td>2.249948e-06</td>
      <td>1.434990e-10</td>
      <td>4.955593e-13</td>
      <td>2.682206e-04</td>
      <td>7.840119e-06</td>
      <td>5.496295e-08</td>
      <td>1.293376e-14</td>
      <td>0.000000e+00</td>
      <td>3.142143e-23</td>
      <td>6.673105e-10</td>
      <td>3.838629e-10</td>
      <td>8.258444e-13</td>
      <td>3.468067e-07</td>
      <td>1.964702e-11</td>
      <td>2.688929e-11</td>
      <td>1.439036e-06</td>
      <td>3.795220e-10</td>
      <td>5.489724e-12</td>
      <td>5.848340e-05</td>
      <td>9.163638e-07</td>
      <td>1.254976e-08</td>
      <td>7.625626e-10</td>
      <td>3.406929e-10</td>
      <td>6.956260e-13</td>
      <td>1.090325e-07</td>
      <td>0.001586</td>
      <td>1.138962e-05</td>
      <td>5.878885e-10</td>
      <td>4.883177e-12</td>
    </tr>
    <tr>
      <th>network_matched_filter_snr_upper</th>
      <td>3.867973e-06</td>
      <td>2.236819e-10</td>
      <td>7.658591e-14</td>
      <td>3.905596e-04</td>
      <td>1.101391e-05</td>
      <td>9.464447e-08</td>
      <td>2.120415e-12</td>
      <td>3.142143e-23</td>
      <td>0.000000e+00</td>
      <td>4.185701e-09</td>
      <td>2.189670e-09</td>
      <td>1.932310e-12</td>
      <td>5.600883e-07</td>
      <td>6.924753e-11</td>
      <td>5.057236e-11</td>
      <td>2.690050e-06</td>
      <td>4.824583e-10</td>
      <td>2.866801e-12</td>
      <td>9.132304e-05</td>
      <td>1.460744e-06</td>
      <td>1.750544e-08</td>
      <td>4.807895e-09</td>
      <td>1.917457e-09</td>
      <td>1.711595e-12</td>
      <td>2.387652e-09</td>
      <td>0.000676</td>
      <td>1.816268e-05</td>
      <td>7.041197e-10</td>
      <td>1.951242e-12</td>
    </tr>
    <tr>
      <th>luminosity_distance</th>
      <td>3.136846e-16</td>
      <td>1.007265e-22</td>
      <td>2.675472e-15</td>
      <td>5.153927e-11</td>
      <td>1.822501e-14</td>
      <td>4.631302e-21</td>
      <td>6.127532e-09</td>
      <td>6.673105e-10</td>
      <td>4.185701e-09</td>
      <td>7.230709e-213</td>
      <td>5.236676e-46</td>
      <td>4.644406e-23</td>
      <td>1.048579e-07</td>
      <td>1.513719e-20</td>
      <td>1.915785e-16</td>
      <td>2.615198e-16</td>
      <td>7.960237e-22</td>
      <td>2.830820e-17</td>
      <td>1.095267e-12</td>
      <td>2.752652e-17</td>
      <td>1.222944e-21</td>
      <td>1.270112e-47</td>
      <td>1.966139e-44</td>
      <td>4.372988e-23</td>
      <td>7.437117e-04</td>
      <td>0.033895</td>
      <td>2.275263e-14</td>
      <td>4.024662e-22</td>
      <td>7.227964e-17</td>
    </tr>
    <tr>
      <th>luminosity_distance_lower</th>
      <td>9.057529e-16</td>
      <td>4.911785e-23</td>
      <td>1.079682e-15</td>
      <td>1.051495e-10</td>
      <td>3.847795e-14</td>
      <td>9.894113e-21</td>
      <td>3.612820e-09</td>
      <td>3.838629e-10</td>
      <td>2.189670e-09</td>
      <td>5.236676e-46</td>
      <td>0.000000e+00</td>
      <td>1.874781e-24</td>
      <td>8.675852e-08</td>
      <td>1.087397e-20</td>
      <td>1.432012e-16</td>
      <td>7.143443e-16</td>
      <td>4.009860e-22</td>
      <td>1.171498e-17</td>
      <td>2.526618e-12</td>
      <td>6.562725e-17</td>
      <td>1.391319e-21</td>
      <td>2.821923e-40</td>
      <td>2.386156e-47</td>
      <td>2.080660e-24</td>
      <td>5.040167e-04</td>
      <td>0.026469</td>
      <td>5.739631e-14</td>
      <td>2.157047e-22</td>
      <td>2.960730e-17</td>
    </tr>
    <tr>
      <th>luminosity_distance_upper</th>
      <td>4.357122e-13</td>
      <td>8.289530e-24</td>
      <td>9.388695e-20</td>
      <td>8.028237e-09</td>
      <td>6.183032e-12</td>
      <td>1.889477e-17</td>
      <td>1.341754e-10</td>
      <td>8.258444e-13</td>
      <td>1.932310e-12</td>
      <td>4.644406e-23</td>
      <td>1.874781e-24</td>
      <td>7.230709e-213</td>
      <td>3.360957e-08</td>
      <td>1.850443e-21</td>
      <td>7.814957e-18</td>
      <td>2.861636e-13</td>
      <td>1.197804e-22</td>
      <td>4.721540e-21</td>
      <td>3.772910e-10</td>
      <td>3.738458e-14</td>
      <td>3.728574e-19</td>
      <td>1.337448e-22</td>
      <td>1.914095e-24</td>
      <td>1.264076e-49</td>
      <td>2.208681e-05</td>
      <td>0.007037</td>
      <td>1.381150e-11</td>
      <td>1.595079e-22</td>
      <td>1.011244e-20</td>
    </tr>
    <tr>
      <th>chi_eff</th>
      <td>4.601379e-05</td>
      <td>2.383769e-07</td>
      <td>1.245116e-07</td>
      <td>1.670918e-03</td>
      <td>1.826611e-04</td>
      <td>6.205122e-06</td>
      <td>1.249804e-06</td>
      <td>3.468067e-07</td>
      <td>5.600883e-07</td>
      <td>1.048579e-07</td>
      <td>8.675852e-08</td>
      <td>3.360957e-08</td>
      <td>0.000000e+00</td>
      <td>3.811511e-06</td>
      <td>1.623952e-05</td>
      <td>6.785248e-05</td>
      <td>1.265358e-06</td>
      <td>4.434390e-07</td>
      <td>4.835460e-04</td>
      <td>3.501432e-05</td>
      <td>4.506714e-06</td>
      <td>1.754188e-07</td>
      <td>1.360661e-07</td>
      <td>5.136965e-08</td>
      <td>5.514387e-04</td>
      <td>0.005168</td>
      <td>1.643364e-04</td>
      <td>1.045957e-06</td>
      <td>3.064566e-07</td>
    </tr>
    <tr>
      <th>chi_eff_lower</th>
      <td>1.406911e-15</td>
      <td>6.253126e-23</td>
      <td>1.263230e-17</td>
      <td>8.876831e-11</td>
      <td>2.020617e-14</td>
      <td>7.489701e-20</td>
      <td>2.737687e-09</td>
      <td>1.964702e-11</td>
      <td>6.924753e-11</td>
      <td>1.513719e-20</td>
      <td>1.087397e-20</td>
      <td>1.850443e-21</td>
      <td>3.811511e-06</td>
      <td>0.000000e+00</td>
      <td>3.503246e-26</td>
      <td>8.037038e-16</td>
      <td>2.330415e-23</td>
      <td>9.521977e-20</td>
      <td>2.656778e-12</td>
      <td>1.304915e-16</td>
      <td>2.153172e-21</td>
      <td>2.899202e-21</td>
      <td>1.015888e-21</td>
      <td>2.043372e-22</td>
      <td>1.666088e-04</td>
      <td>0.044161</td>
      <td>5.591434e-14</td>
      <td>1.699320e-23</td>
      <td>2.652554e-19</td>
    </tr>
    <tr>
      <th>chi_eff_upper</th>
      <td>1.812327e-14</td>
      <td>4.392393e-20</td>
      <td>3.071341e-18</td>
      <td>2.147474e-10</td>
      <td>6.089007e-14</td>
      <td>1.312369e-17</td>
      <td>4.615810e-09</td>
      <td>2.688929e-11</td>
      <td>5.057236e-11</td>
      <td>1.915785e-16</td>
      <td>1.432012e-16</td>
      <td>7.814957e-18</td>
      <td>1.623952e-05</td>
      <td>3.503246e-26</td>
      <td>7.230709e-213</td>
      <td>9.867745e-15</td>
      <td>1.256422e-20</td>
      <td>5.048458e-20</td>
      <td>9.730297e-12</td>
      <td>1.440978e-15</td>
      <td>5.089606e-19</td>
      <td>5.932545e-17</td>
      <td>2.696547e-17</td>
      <td>1.453767e-18</td>
      <td>1.903353e-04</td>
      <td>NaN</td>
      <td>3.429017e-13</td>
      <td>1.171875e-20</td>
      <td>9.215536e-20</td>
    </tr>
    <tr>
      <th>total_mass_source</th>
      <td>7.722088e-33</td>
      <td>3.753790e-17</td>
      <td>1.212501e-11</td>
      <td>5.230305e-20</td>
      <td>2.842232e-29</td>
      <td>2.437694e-25</td>
      <td>1.569187e-05</td>
      <td>1.439036e-06</td>
      <td>2.690050e-06</td>
      <td>2.615198e-16</td>
      <td>7.143443e-16</td>
      <td>2.861636e-13</td>
      <td>6.785248e-05</td>
      <td>8.037038e-16</td>
      <td>9.867745e-15</td>
      <td>0.000000e+00</td>
      <td>8.718207e-19</td>
      <td>9.776448e-14</td>
      <td>3.810319e-24</td>
      <td>5.460683e-32</td>
      <td>1.602768e-22</td>
      <td>8.232555e-17</td>
      <td>2.861371e-16</td>
      <td>1.730558e-13</td>
      <td>1.260608e-02</td>
      <td>NaN</td>
      <td>2.296272e-29</td>
      <td>8.022102e-19</td>
      <td>3.429746e-13</td>
    </tr>
    <tr>
      <th>total_mass_source_lower</th>
      <td>3.368283e-18</td>
      <td>5.474945e-32</td>
      <td>4.895309e-19</td>
      <td>6.811159e-12</td>
      <td>1.206024e-16</td>
      <td>2.268857e-25</td>
      <td>3.001843e-08</td>
      <td>3.795220e-10</td>
      <td>4.824583e-10</td>
      <td>7.960237e-22</td>
      <td>4.009860e-22</td>
      <td>1.197804e-22</td>
      <td>1.265358e-06</td>
      <td>2.330415e-23</td>
      <td>1.256422e-20</td>
      <td>8.718207e-19</td>
      <td>0.000000e+00</td>
      <td>4.183691e-23</td>
      <td>9.380646e-14</td>
      <td>3.628998e-20</td>
      <td>4.903926e-30</td>
      <td>3.303447e-22</td>
      <td>7.841804e-23</td>
      <td>2.964920e-23</td>
      <td>2.902308e-04</td>
      <td>0.026712</td>
      <td>6.615027e-16</td>
      <td>7.907930e-43</td>
      <td>7.386841e-22</td>
    </tr>
    <tr>
      <th>total_mass_source_upper</th>
      <td>1.532938e-13</td>
      <td>7.705030e-25</td>
      <td>1.388283e-30</td>
      <td>3.427911e-09</td>
      <td>1.453581e-12</td>
      <td>6.163557e-17</td>
      <td>3.837445e-09</td>
      <td>5.489724e-12</td>
      <td>2.866801e-12</td>
      <td>2.830820e-17</td>
      <td>1.171498e-17</td>
      <td>4.721540e-21</td>
      <td>4.434390e-07</td>
      <td>9.521977e-20</td>
      <td>5.048458e-20</td>
      <td>9.776448e-14</td>
      <td>4.183691e-23</td>
      <td>0.000000e+00</td>
      <td>1.526926e-10</td>
      <td>1.052778e-14</td>
      <td>5.305639e-19</td>
      <td>2.508566e-17</td>
      <td>5.416365e-18</td>
      <td>1.783134e-21</td>
      <td>1.157030e-04</td>
      <td>0.012132</td>
      <td>5.231721e-12</td>
      <td>8.250429e-23</td>
      <td>4.260012e-42</td>
    </tr>
    <tr>
      <th>chirp_mass_source</th>
      <td>5.192259e-26</td>
      <td>5.886118e-13</td>
      <td>4.210593e-09</td>
      <td>1.031650e-32</td>
      <td>7.009607e-29</td>
      <td>3.022607e-18</td>
      <td>2.758899e-04</td>
      <td>5.848340e-05</td>
      <td>9.132304e-05</td>
      <td>1.095267e-12</td>
      <td>2.526618e-12</td>
      <td>3.772910e-10</td>
      <td>4.835460e-04</td>
      <td>2.656778e-12</td>
      <td>9.730297e-12</td>
      <td>3.810319e-24</td>
      <td>9.380646e-14</td>
      <td>1.526926e-10</td>
      <td>8.376948e-209</td>
      <td>6.981041e-23</td>
      <td>1.888034e-16</td>
      <td>4.763282e-13</td>
      <td>1.370488e-12</td>
      <td>2.657937e-10</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.649392e-34</td>
      <td>5.217599e-14</td>
      <td>2.945969e-10</td>
    </tr>
    <tr>
      <th>chirp_mass_source_lower</th>
      <td>1.630534e-33</td>
      <td>1.168080e-18</td>
      <td>1.565564e-12</td>
      <td>6.039683e-19</td>
      <td>8.853396e-31</td>
      <td>1.367952e-29</td>
      <td>6.699207e-06</td>
      <td>9.163638e-07</td>
      <td>1.460744e-06</td>
      <td>2.752652e-17</td>
      <td>6.562725e-17</td>
      <td>3.738458e-14</td>
      <td>3.501432e-05</td>
      <td>1.304915e-16</td>
      <td>1.440978e-15</td>
      <td>5.460683e-32</td>
      <td>3.628998e-20</td>
      <td>1.052778e-14</td>
      <td>6.981041e-23</td>
      <td>7.230709e-213</td>
      <td>1.164092e-25</td>
      <td>8.693180e-18</td>
      <td>2.471273e-17</td>
      <td>2.248853e-14</td>
      <td>9.049294e-03</td>
      <td>NaN</td>
      <td>6.134334e-28</td>
      <td>1.301647e-20</td>
      <td>3.070662e-14</td>
    </tr>
    <tr>
      <th>chirp_mass_source_upper</th>
      <td>2.890195e-22</td>
      <td>6.214730e-26</td>
      <td>5.386165e-16</td>
      <td>4.198612e-14</td>
      <td>2.491595e-20</td>
      <td>3.959121e-36</td>
      <td>2.374753e-07</td>
      <td>1.254976e-08</td>
      <td>1.750544e-08</td>
      <td>1.222944e-21</td>
      <td>1.391319e-21</td>
      <td>3.728574e-19</td>
      <td>4.506714e-06</td>
      <td>2.153172e-21</td>
      <td>5.089606e-19</td>
      <td>1.602768e-22</td>
      <td>4.903926e-30</td>
      <td>5.305639e-19</td>
      <td>1.888034e-16</td>
      <td>1.164092e-25</td>
      <td>0.000000e+00</td>
      <td>3.086507e-22</td>
      <td>2.654492e-22</td>
      <td>1.596989e-19</td>
      <td>1.097560e-03</td>
      <td>NaN</td>
      <td>3.133565e-19</td>
      <td>2.095692e-31</td>
      <td>2.631119e-18</td>
    </tr>
    <tr>
      <th>redshift</th>
      <td>1.037333e-16</td>
      <td>6.433552e-23</td>
      <td>2.843388e-15</td>
      <td>2.524420e-11</td>
      <td>6.805960e-15</td>
      <td>1.053057e-21</td>
      <td>7.580384e-09</td>
      <td>7.625626e-10</td>
      <td>4.807895e-09</td>
      <td>1.270112e-47</td>
      <td>2.821923e-40</td>
      <td>1.337448e-22</td>
      <td>1.754188e-07</td>
      <td>2.899202e-21</td>
      <td>5.932545e-17</td>
      <td>8.232555e-17</td>
      <td>3.303447e-22</td>
      <td>2.508566e-17</td>
      <td>4.763282e-13</td>
      <td>8.693180e-18</td>
      <td>3.086507e-22</td>
      <td>0.000000e+00</td>
      <td>1.201425e-43</td>
      <td>9.316592e-23</td>
      <td>8.366689e-04</td>
      <td>0.040483</td>
      <td>8.579264e-15</td>
      <td>1.566297e-22</td>
      <td>6.696124e-17</td>
    </tr>
    <tr>
      <th>redshift_lower</th>
      <td>3.820235e-16</td>
      <td>1.256152e-23</td>
      <td>6.506532e-16</td>
      <td>6.254192e-11</td>
      <td>1.705361e-14</td>
      <td>2.388793e-21</td>
      <td>3.771030e-09</td>
      <td>3.406929e-10</td>
      <td>1.917457e-09</td>
      <td>1.966139e-44</td>
      <td>2.386156e-47</td>
      <td>1.914095e-24</td>
      <td>1.360661e-07</td>
      <td>1.015888e-21</td>
      <td>2.696547e-17</td>
      <td>2.861371e-16</td>
      <td>7.841804e-23</td>
      <td>5.416365e-18</td>
      <td>1.370488e-12</td>
      <td>2.471273e-17</td>
      <td>2.654492e-22</td>
      <td>1.201425e-43</td>
      <td>0.000000e+00</td>
      <td>1.375399e-24</td>
      <td>5.135346e-04</td>
      <td>0.030439</td>
      <td>2.729509e-14</td>
      <td>3.870670e-23</td>
      <td>1.444684e-17</td>
    </tr>
    <tr>
      <th>redshift_upper</th>
      <td>2.758241e-13</td>
      <td>2.525131e-24</td>
      <td>5.066937e-20</td>
      <td>5.890683e-09</td>
      <td>3.917765e-12</td>
      <td>9.751849e-18</td>
      <td>1.348714e-10</td>
      <td>6.956260e-13</td>
      <td>1.711595e-12</td>
      <td>4.372988e-23</td>
      <td>2.080660e-24</td>
      <td>1.264076e-49</td>
      <td>5.136965e-08</td>
      <td>2.043372e-22</td>
      <td>1.453767e-18</td>
      <td>1.730558e-13</td>
      <td>2.964920e-23</td>
      <td>1.783134e-21</td>
      <td>2.657937e-10</td>
      <td>2.248853e-14</td>
      <td>1.596989e-19</td>
      <td>9.316592e-23</td>
      <td>1.375399e-24</td>
      <td>7.230709e-213</td>
      <td>2.361499e-05</td>
      <td>0.008143</td>
      <td>9.162416e-12</td>
      <td>4.200791e-23</td>
      <td>4.223580e-21</td>
    </tr>
    <tr>
      <th>far</th>
      <td>1.549977e-02</td>
      <td>3.388679e-04</td>
      <td>3.434831e-05</td>
      <td>NaN</td>
      <td>2.002270e-02</td>
      <td>2.338337e-03</td>
      <td>1.020624e-06</td>
      <td>1.090325e-07</td>
      <td>2.387652e-09</td>
      <td>7.437117e-04</td>
      <td>5.040167e-04</td>
      <td>2.208681e-05</td>
      <td>5.514387e-04</td>
      <td>1.666088e-04</td>
      <td>1.903353e-04</td>
      <td>1.260608e-02</td>
      <td>2.902308e-04</td>
      <td>1.157030e-04</td>
      <td>NaN</td>
      <td>9.049294e-03</td>
      <td>1.097560e-03</td>
      <td>8.366689e-04</td>
      <td>5.135346e-04</td>
      <td>2.361499e-05</td>
      <td>0.000000e+00</td>
      <td>0.000010</td>
      <td>2.788954e-02</td>
      <td>3.348846e-04</td>
      <td>1.010219e-04</td>
    </tr>
    <tr>
      <th>final_mass_source</th>
      <td>3.508650e-34</td>
      <td>5.845433e-15</td>
      <td>2.408078e-10</td>
      <td>2.892281e-25</td>
      <td>2.699745e-34</td>
      <td>1.343855e-21</td>
      <td>7.768916e-05</td>
      <td>1.138962e-05</td>
      <td>1.816268e-05</td>
      <td>2.275263e-14</td>
      <td>5.739631e-14</td>
      <td>1.381150e-11</td>
      <td>1.643364e-04</td>
      <td>5.591434e-14</td>
      <td>3.429017e-13</td>
      <td>2.296272e-29</td>
      <td>6.615027e-16</td>
      <td>5.231721e-12</td>
      <td>8.649392e-34</td>
      <td>6.134334e-28</td>
      <td>3.133565e-19</td>
      <td>8.579264e-15</td>
      <td>2.729509e-14</td>
      <td>9.162416e-12</td>
      <td>2.788954e-02</td>
      <td>NaN</td>
      <td>0.000000e+00</td>
      <td>3.144979e-16</td>
      <td>1.142456e-11</td>
    </tr>
    <tr>
      <th>final_mass_source_lower</th>
      <td>1.231781e-18</td>
      <td>6.873241e-34</td>
      <td>4.748794e-19</td>
      <td>4.349177e-12</td>
      <td>5.957291e-17</td>
      <td>2.689311e-26</td>
      <td>3.594926e-08</td>
      <td>5.878885e-10</td>
      <td>7.041197e-10</td>
      <td>4.024662e-22</td>
      <td>2.157047e-22</td>
      <td>1.595079e-22</td>
      <td>1.045957e-06</td>
      <td>1.699320e-23</td>
      <td>1.171875e-20</td>
      <td>8.022102e-19</td>
      <td>7.907930e-43</td>
      <td>8.250429e-23</td>
      <td>5.217599e-14</td>
      <td>1.301647e-20</td>
      <td>2.095692e-31</td>
      <td>1.566297e-22</td>
      <td>3.870670e-23</td>
      <td>4.200791e-23</td>
      <td>3.348846e-04</td>
      <td>0.029347</td>
      <td>3.144979e-16</td>
      <td>7.230709e-213</td>
      <td>7.024697e-22</td>
    </tr>
    <tr>
      <th>final_mass_source_upper</th>
      <td>3.806204e-13</td>
      <td>5.157248e-24</td>
      <td>4.057670e-34</td>
      <td>6.051949e-09</td>
      <td>3.324090e-12</td>
      <td>2.233381e-16</td>
      <td>3.101063e-09</td>
      <td>4.883177e-12</td>
      <td>1.951242e-12</td>
      <td>7.227964e-17</td>
      <td>2.960730e-17</td>
      <td>1.011244e-20</td>
      <td>3.064566e-07</td>
      <td>2.652554e-19</td>
      <td>9.215536e-20</td>
      <td>3.429746e-13</td>
      <td>7.386841e-22</td>
      <td>4.260012e-42</td>
      <td>2.945969e-10</td>
      <td>3.070662e-14</td>
      <td>2.631119e-18</td>
      <td>6.696124e-17</td>
      <td>1.444684e-17</td>
      <td>4.223580e-21</td>
      <td>1.010219e-04</td>
      <td>0.011519</td>
      <td>1.142456e-11</td>
      <td>7.024697e-22</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>p_astro</th>
      <td>NaN</td>
      <td>2.575412e-02</td>
      <td>7.280247e-03</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.199252e-04</td>
      <td>1.586420e-03</td>
      <td>6.761854e-04</td>
      <td>3.389481e-02</td>
      <td>2.646942e-02</td>
      <td>7.037256e-03</td>
      <td>5.167684e-03</td>
      <td>4.416088e-02</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.671188e-02</td>
      <td>1.213166e-02</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.048327e-02</td>
      <td>3.043913e-02</td>
      <td>8.143217e-03</td>
      <td>9.710429e-06</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>2.934673e-02</td>
      <td>1.151923e-02</td>
    </tr>
  </tbody>
</table>
</div>



After the first threshhold, it still displays most of the data as many appear to be statistically significant so the threshold was lowered even further and, as the correlation is obvious, uncertainties were dropped as well.


```python
gP = onlySignificant(pearsonDf, 0.001)
droppable = ['mass_1_source_lower', 'mass_1_source_upper',
       'mass_2_source_lower', 'mass_2_source_upper','network_matched_filter_snr_lower',
       'network_matched_filter_snr_upper','luminosity_distance_lower', 'luminosity_distance_upper',
       'chi_eff_lower', 'chi_eff_upper','total_mass_source_lower', 'total_mass_source_upper',
       'chirp_mass_source_lower','chirp_mass_source_upper', 'redshift_lower',
       'redshift_upper','final_mass_source_lower', 'final_mass_source_upper']
gPdf = pd.DataFrame.from_dict(gP)
revised = gPdf.drop(droppable, axis = 1)
final = revised.drop(droppable, axis = 0)
final
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mass_1_source</th>
      <th>mass_2_source</th>
      <th>network_matched_filter_snr</th>
      <th>luminosity_distance</th>
      <th>chi_eff</th>
      <th>total_mass_source</th>
      <th>chirp_mass_source</th>
      <th>redshift</th>
      <th>far</th>
      <th>p_astro</th>
      <th>final_mass_source</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mass_1_source</th>
      <td>0.000000e+00</td>
      <td>7.681062e-21</td>
      <td>2.186359e-05</td>
      <td>3.136846e-16</td>
      <td>4.601379e-05</td>
      <td>7.722088e-33</td>
      <td>5.192259e-26</td>
      <td>1.037333e-16</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.508650e-34</td>
    </tr>
    <tr>
      <th>mass_2_source</th>
      <td>7.681062e-21</td>
      <td>7.230709e-213</td>
      <td>9.768584e-04</td>
      <td>5.153927e-11</td>
      <td>NaN</td>
      <td>5.230305e-20</td>
      <td>1.031650e-32</td>
      <td>2.524420e-11</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.892281e-25</td>
    </tr>
    <tr>
      <th>network_matched_filter_snr</th>
      <td>2.186359e-05</td>
      <td>9.768584e-04</td>
      <td>7.230709e-213</td>
      <td>6.127532e-09</td>
      <td>1.249804e-06</td>
      <td>1.569187e-05</td>
      <td>2.758899e-04</td>
      <td>7.580384e-09</td>
      <td>0.000001</td>
      <td>0.00012</td>
      <td>7.768916e-05</td>
    </tr>
    <tr>
      <th>luminosity_distance</th>
      <td>3.136846e-16</td>
      <td>5.153927e-11</td>
      <td>6.127532e-09</td>
      <td>7.230709e-213</td>
      <td>1.048579e-07</td>
      <td>2.615198e-16</td>
      <td>1.095267e-12</td>
      <td>1.270112e-47</td>
      <td>0.000744</td>
      <td>NaN</td>
      <td>2.275263e-14</td>
    </tr>
    <tr>
      <th>chi_eff</th>
      <td>4.601379e-05</td>
      <td>NaN</td>
      <td>1.249804e-06</td>
      <td>1.048579e-07</td>
      <td>0.000000e+00</td>
      <td>6.785248e-05</td>
      <td>4.835460e-04</td>
      <td>1.754188e-07</td>
      <td>0.000551</td>
      <td>NaN</td>
      <td>1.643364e-04</td>
    </tr>
    <tr>
      <th>total_mass_source</th>
      <td>7.722088e-33</td>
      <td>5.230305e-20</td>
      <td>1.569187e-05</td>
      <td>2.615198e-16</td>
      <td>6.785248e-05</td>
      <td>0.000000e+00</td>
      <td>3.810319e-24</td>
      <td>8.232555e-17</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.296272e-29</td>
    </tr>
    <tr>
      <th>chirp_mass_source</th>
      <td>5.192259e-26</td>
      <td>1.031650e-32</td>
      <td>2.758899e-04</td>
      <td>1.095267e-12</td>
      <td>4.835460e-04</td>
      <td>3.810319e-24</td>
      <td>8.376948e-209</td>
      <td>4.763282e-13</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.649392e-34</td>
    </tr>
    <tr>
      <th>redshift</th>
      <td>1.037333e-16</td>
      <td>2.524420e-11</td>
      <td>7.580384e-09</td>
      <td>1.270112e-47</td>
      <td>1.754188e-07</td>
      <td>8.232555e-17</td>
      <td>4.763282e-13</td>
      <td>0.000000e+00</td>
      <td>0.000837</td>
      <td>NaN</td>
      <td>8.579264e-15</td>
    </tr>
    <tr>
      <th>final_mass_source</th>
      <td>3.508650e-34</td>
      <td>2.892281e-25</td>
      <td>7.768916e-05</td>
      <td>2.275263e-14</td>
      <td>1.643364e-04</td>
      <td>2.296272e-29</td>
      <td>8.649392e-34</td>
      <td>8.579264e-15</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>far</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.020624e-06</td>
      <td>7.437117e-04</td>
      <td>5.514387e-04</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.366689e-04</td>
      <td>0.000000</td>
      <td>0.00001</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>p_astro</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.199252e-04</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000010</td>
      <td>0.00000</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



At this threshhold only very strong correlations are present, so before reducing the threshhold even further, some analysis should be done. An interesting variable, p_astro (probabiltiy of astrophysical origin) is only strongly correlated with far ($yr^{-1}$) and network_matched_filter_snr (signal to nosie ratio). p_astro is a threshhold probability used in the search algorithm [3] looking for potentional gravitational wave signals. This was used in conjunction with the FAR and the SNR to narrow down the search of candidates. This is likely why these three variables are so heavily correlated.

More notably was the effective spin $\chi_{eff}$ variable. Curiously it is more dependent on mass 1 than mass 2 as it is more likely to be correlated with mass 1. While it is irrelevent that it is more correlated with mass 1 than mass 2 as they could be interchanged with little effect, mass 1 is always the larger of the two masses in the system, likely due to a convention set by LIGO. Thus, what this correlation is really showing is that effective spin is more strongly dependent on the larger of the two objects. That is to say that the more massive object in a binary system has a stronger effect on the events effective spin as a whole than does the less massive object.

[3] R. Abbott et al. (LIGO Scientific Collaboration and Virgo Collaboration), "Open data from the first and second observing runs of Advanced LIGO and Advanced Virgo"


```python
gP = onlySignificant(pearsonDf, 0.0001)
gPdf = pd.DataFrame.from_dict(gP)
droppable = ['mass_1_source_lower', 'mass_1_source_upper',
       'mass_2_source_lower', 'mass_2_source_upper',
       'network_matched_filter_snr_lower',
       'network_matched_filter_snr_upper',
       'luminosity_distance_lower', 'luminosity_distance_upper',
       'chi_eff_lower', 'chi_eff_upper',
       'total_mass_source_lower', 'total_mass_source_upper',
       'chirp_mass_source_lower',
       'chirp_mass_source_upper','redshift_lower',
       'redshift_upper', 'p_astro',
       'final_mass_source_lower', 'final_mass_source_upper']
revised = gPdf.drop(droppable, axis = 1)
final = revised.drop(droppable, axis = 0)
final
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mass_1_source</th>
      <th>mass_2_source</th>
      <th>network_matched_filter_snr</th>
      <th>luminosity_distance</th>
      <th>chi_eff</th>
      <th>total_mass_source</th>
      <th>chirp_mass_source</th>
      <th>redshift</th>
      <th>far</th>
      <th>final_mass_source</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mass_1_source</th>
      <td>0.000000e+00</td>
      <td>7.681062e-21</td>
      <td>2.186359e-05</td>
      <td>3.136846e-16</td>
      <td>4.601379e-05</td>
      <td>7.722088e-33</td>
      <td>5.192259e-26</td>
      <td>1.037333e-16</td>
      <td>NaN</td>
      <td>3.508650e-34</td>
    </tr>
    <tr>
      <th>mass_2_source</th>
      <td>7.681062e-21</td>
      <td>7.230709e-213</td>
      <td>NaN</td>
      <td>5.153927e-11</td>
      <td>NaN</td>
      <td>5.230305e-20</td>
      <td>1.031650e-32</td>
      <td>2.524420e-11</td>
      <td>NaN</td>
      <td>2.892281e-25</td>
    </tr>
    <tr>
      <th>network_matched_filter_snr</th>
      <td>2.186359e-05</td>
      <td>NaN</td>
      <td>7.230709e-213</td>
      <td>6.127532e-09</td>
      <td>1.249804e-06</td>
      <td>1.569187e-05</td>
      <td>NaN</td>
      <td>7.580384e-09</td>
      <td>0.000001</td>
      <td>7.768916e-05</td>
    </tr>
    <tr>
      <th>luminosity_distance</th>
      <td>3.136846e-16</td>
      <td>5.153927e-11</td>
      <td>6.127532e-09</td>
      <td>7.230709e-213</td>
      <td>1.048579e-07</td>
      <td>2.615198e-16</td>
      <td>1.095267e-12</td>
      <td>1.270112e-47</td>
      <td>NaN</td>
      <td>2.275263e-14</td>
    </tr>
    <tr>
      <th>chi_eff</th>
      <td>4.601379e-05</td>
      <td>NaN</td>
      <td>1.249804e-06</td>
      <td>1.048579e-07</td>
      <td>0.000000e+00</td>
      <td>6.785248e-05</td>
      <td>NaN</td>
      <td>1.754188e-07</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>total_mass_source</th>
      <td>7.722088e-33</td>
      <td>5.230305e-20</td>
      <td>1.569187e-05</td>
      <td>2.615198e-16</td>
      <td>6.785248e-05</td>
      <td>0.000000e+00</td>
      <td>3.810319e-24</td>
      <td>8.232555e-17</td>
      <td>NaN</td>
      <td>2.296272e-29</td>
    </tr>
    <tr>
      <th>chirp_mass_source</th>
      <td>5.192259e-26</td>
      <td>1.031650e-32</td>
      <td>NaN</td>
      <td>1.095267e-12</td>
      <td>NaN</td>
      <td>3.810319e-24</td>
      <td>8.376948e-209</td>
      <td>4.763282e-13</td>
      <td>NaN</td>
      <td>8.649392e-34</td>
    </tr>
    <tr>
      <th>redshift</th>
      <td>1.037333e-16</td>
      <td>2.524420e-11</td>
      <td>7.580384e-09</td>
      <td>1.270112e-47</td>
      <td>1.754188e-07</td>
      <td>8.232555e-17</td>
      <td>4.763282e-13</td>
      <td>0.000000e+00</td>
      <td>NaN</td>
      <td>8.579264e-15</td>
    </tr>
    <tr>
      <th>final_mass_source</th>
      <td>3.508650e-34</td>
      <td>2.892281e-25</td>
      <td>7.768916e-05</td>
      <td>2.275263e-14</td>
      <td>NaN</td>
      <td>2.296272e-29</td>
      <td>8.649392e-34</td>
      <td>8.579264e-15</td>
      <td>NaN</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>far</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.020624e-06</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
final.replace(np.NaN, 0)
to_graph = []
ligo = pd.read_csv("Ligo.csv")
for i in final.columns:
    for j in final.columns:
        if final[i][j] != 0:
            if (j, i) not in to_graph:
                if (i != j):
                    to_graph.append((i,j))
fig = plt.figure(figsize=(10,60))
for i in range(len(to_graph)):
    plt.subplot(23, 2, i+1)
    
    plt.scatter(ligo[to_graph[i][0]], ligo[to_graph[i][1]])
    plt.title(f'{to_graph[i][1]} vs {to_graph[i][0]}')
    if(to_graph[i][0] == "far"):
        plt.xscale("log")
    if(to_graph[i][1] == "far"):
        plt.yscale("log")
plt.subplots_adjust(hspace=0.5)

```


    
![png](README_files/README_57_0.png)
    


Analyzing these graphs a few anomalies stand out. Firstly, the peculiarity of various relationships with far. It appears that a number of the events were all using the same far value in the search algorithm. Perhaps they are from the same region of space? More likely is that that far value (somewhere around $10^{-5} yr^{-1}$) is characteristic of a specific type of merger, i.e (black hole - black hole, neutron star -neutron-star, etc.) 

More interesting correlations, that were not immediately obvious were when signal to noise ratio was compared to redshift, luminosity distance and total mass. The strongest of these was found in luminosity distance, however the correlation can be seen through the other two variables as well. Since the strongest correlation was between luminosity distance and SNR this relationship is focused on. The next step was to fit a non-linear model to this data.


```python
plt.scatter(ligo["network_matched_filter_snr"], ligo["luminosity_distance"])
x = ligo["network_matched_filter_snr"].to_numpy()
plt.title("Luminosity Distance vs Signal to Noise Ratio")
plt.ylabel(f'Luminosity Distance ({u.Gpc})')
plt.xlabel("SNR")
```




    Text(0.5, 0, 'SNR')




    
![png](README_files/README_60_1.png)
    


Although achieved through a significant amount of trial and error, the best model was $7000e^{-1x} + 0x$. To further cement the correlation (and for my own enjoyment), a machine learning algorithm was used to predict points based on the data. Although the outcome mainly shows that the model is indeed a decent fit, running the algorithm on a larger data set may prove that the model is inaccurate at extrema and edge cases. 


```python
# From Lecture 28
def f_nonlinear(x,A=1,B=2,C=3,addnoise=False,noise=1.0,plot=False):
    if addnoise:
        noise = np.random.normal(scale=noise,size=len(x))
    else:
        noise = 0
    y = A*np.exp(-B*x) + C*x + noise

    if plot:
        plt.scatter(x,y)
    return y

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# x = ligo.loc[ligo[].notna()].to_numpy()
x = ligo.loc[ligo['luminosity_distance'].notna(), 'network_matched_filter_snr'].to_numpy()
y = ligo.loc[ligo['luminosity_distance'].notna(), 'luminosity_distance'].to_numpy()


x_train, x_test, y_train, y_test =\
 train_test_split(x, y, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=5, random_state=0)
regr.fit(x_train.reshape(-1, 1), y_train)
pred = regr.predict(x_test.reshape(-1, 1))

plt.figure(figsize=(12,6))
plt.scatter(x_train,y_train,label="Training")
plt.scatter(x_test,pred,color='r',label="Testing")
x.sort()
plt.plot(x,f_nonlinear(x,A=7000,B=0.1, C=0),color='k', label=r"$7000e^{-.1x} + 0x$");

plt.ylim(-100, 10000)
plt.legend();

```


    
![png](README_files/README_62_0.png)
    


What is the significance of the strong correlation between SNR and luminosity distance? SNR measures the strength of the signal vs the noise, so at lower SNR values the signal is harder to decern from the nosie. A negative correlation to luminosity distance makes sense. As the source of a signal gets farther away it will encounter more interference from gas clouds, galaxies and any other objects between the source and Earth. What was unexpected was a non-linear correlation. Looking at the equation relating flux, luminosity, and distance it becomes more apparent:
$$
f = \frac{L}{4\pi d^{2}}
$$

As the distance between the Earth and a source object increases, the flux falls off proportional to $d^{2}$ thus it makes sense that the SNR ratio decreases exponentially when the distance increases. The signal strength is proportional to the distance between the Earth and the signal source. In essence this is a convoluted way of showing the exponential relationship between flux and distance.  

<H2 align="center">Acknowledgements<H2>

Thank you Professor Darling for the general help in answering my questions during tutorial

<H3 align=center>References</H3>

[1] LIGO Scientific Collaboration, B. P. Abbott et al. (2016). "Observation of Gravitational Waves from a Binary Black Hole Merger". Physical Review Letters. 116 (6): 061102. \
[2] What is an Interferometer? (n.d.). LIGO Lab | Caltech \
[3] Dong, Subo; et al. (2015). "ASASSN-15lh: A highly super-luminous supernova". Science. 351 (6276): 257–260. \
[4] R. Abbott et al. (LIGO Scientific Collaboration and Virgo Collaboration), "Open data from the first and second observing runs of Advanced LIGO and Advanced Virgo", SoftwareX 13 (2021) 100658.


