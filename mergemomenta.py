import averager as avg
import os

#avg.momenta("../2pt/", ["twop_momforsmear_+0_+0_-1", "twop_momforsmear_+0_+0_+1"], "one")
#avg.momenta("../2pt/", ["twop_momforsmear_+0_+0_+0"], "zero")
#avg.momenta("../2pt/", ["twop_momforsmear_+0_+0_-2", "twop_momforsmear_+0_+0_+2"], "two")
#avg.momenta("../2pt/", ["twop_momforsmear_+0_+0_-3", "twop_momforsmear_+0_+0_+3"], "three")
#avg.momenta("../2pt/", ["twop_momforsmear_+0_+0_-4", "twop_momforsmear_+0_+0_+4"], "four")
#avg.momenta("../2pt/", ["twop_momforsmear_+0_+0_-5", "twop_momforsmear_+0_+0_+5"], "five")
path = "momentum-energy/"
files = os.listdir(path=path)
avg.fold(files, path=path)
