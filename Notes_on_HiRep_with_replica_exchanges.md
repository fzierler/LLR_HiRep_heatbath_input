These notes are written from the commit `84a62bd682c07cd0784b242c7edaed790ff075aa` for HiRep with the LLR using the HMC updating algorithm. It should also be applicable to the LLR heatbath code, and reflect the general requirements of using the replica exchange method in HiRep.

The code needs a different setup with and without replica exchanges. The version with replica exchange requires a different setup and naming convention for the input files. 

1. **Without replica exchange**:
- The default flags in `Make/MkFlags` are already in place assuming you are using gcc to compile. The executable can be compiled by simply executing `make` in the `LLR_HMC/` directory.
	```
	cd LLR_HMC/
	make 
	```
- A sample input file for a $4^4$ is already present in the directory in `LLR_HMC/`. You can make sure that the code compiles by executing
	```
	./llr_hmc -i input_hmc_llr -o output_llr_hmc
	```

2. **With replica exchange**
- This codes requires special attention. It can be enabled by adding the following flags to the `Make/MkFlags` file
	```
	MACRO += -DWITH_MPI
	MACRO += -DWITH_UMBRELLA
	```
- Both flags need to be added. Furthermore, the MPI compiler interface needs to be specified. For my local setup I use
	```
	CC = mpicc
	```
- Following those changes, the code can be compiled as before. Note, however, that we cannot run the code as before. This also makes sense, when considering the structure of the code and input files.

3. **Structure of the input files**
- Every replica needs its own input file. A single input file contains only one set of simulation parameters. For the LLR-HMC code, every replica is restricted to a different central energy. In this case, this is done using the variables
	```
	llr:dS = 20.
	llr:S0 = 1020.
	```
- These values need to be set for every replica.
- In principle, the code allows using more than one CPU core for every replica. These can be set using the usual input parameters
	```
	NP_T = 1
	NP_X = 1
	NP_Y = 1
	NP_Z = 1
	```
- If all those values are set to one, HiRep will issue a warning that the MPI code should not be used. This can be safely ignored in this use case, because MPI is still needed to swap replicas if needed.

4. **Notes on the setup for replica exchanges**
The required directory structure is hard-coded in `setup_replicas()` (starting on line 92 in `LibHR/Geometry/process_init.c`). Specifically, it requires that every replica has a separate directory according to
	```
	sprintf(sbuf,"Rep_%d",RID);
	mpiret = chdir(sbuf);
	```
- The input file, output file, and error file will be located relative to the replica directory. They can be specified via the CLI. If omitted they default to
	```
	char input_filename[256] = "input_file";
	char output_filename[256] = "out_0";
	char error_filename[256] = "err_0";
	```
- Note, that the replica-specific parameters are not set by HiRep itself. They need to be set-up manually or by a different helper script.
- Additionally, an input file with the global input parameters needs to be provided. This affects all information that is required before `setup_replicas()` is called. An exemplary file structure would look like
	```
	├── input_file
	├── llr_hmc
	├── Rep_0
	│   └── input_file
	└── Rep_1
	    └── input_file
	```
Note, that the name of the input files within the `Rep_XX/` directories and the main input file at the top level have to match. The global variables specifying the lattice size, MPI partitioning and the number of replicas can be omitted from the input files in `Rep_XX/`. 

Currently, I am hitting a division by zero error, but I can confirm that this setup also works with more than one MPI process per replica. 