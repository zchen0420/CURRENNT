// nc-standardize: standardize inputs and maybe output features in nc file
// used for network training etc.

// compile with: g++ nc-standardize.cpp -lnetcdf -onc-standardize

#include "netcdf.h"
#include <iostream>
#include <sstream>
#include <set>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <unistd.h>
#include <limits>


using namespace std;

int createDimIfNotExists(int ncid, const char* dimName, int* dimid, int dimval)
{
    int ret;

    // try to retrieve specified dimension
    ret = nc_inq_dimid(ncid, dimName, dimid);

    if (ret != NC_NOERR) {
        // dim not found, try to create it
        // enter definition mode
        if ((ret = nc_redef(ncid)) != NC_NOERR) {
            return ret;
        }
        // define variable
        if ((ret = nc_def_dim(ncid, dimName, dimval, dimid)) != NC_NOERR) {
            return ret;
        }
        // exit definition mode
        if ((ret = nc_enddef(ncid)) != NC_NOERR) {
            return ret;
        }
    }else{
//         cout << "ERROR dim exist" <<endl;
    }

    return 0;
}

int createVarIfNotExists(int ncid, const char* varName, const char* dimName, int* varid, float* data)
{
    int ret;

    // try to retrieve specified dimension
    int dimid;
    size_t N;
    if ((ret = nc_inq_dimid(ncid, dimName, &dimid)) != NC_NOERR) {
        return ret;
    }
    if ((ret = nc_inq_dimlen(ncid, dimid, &N)) != NC_NOERR) {
        return ret;
    }

    size_t start[] = {0};
    size_t count[] = {N};

    // try to retrieve specified variable
    ret = nc_inq_varid(ncid, varName, varid);

    if (ret != NC_NOERR) {
        // variable not found, try to create it
        // enter definition mode
        if ((ret = nc_redef(ncid)) != NC_NOERR) {
            return ret;
        }
        // define variable
        if ((ret = nc_def_var(ncid, varName, NC_FLOAT, 1, &dimid, varid)) != NC_NOERR) {
            return ret;
        }
        // exit definition mode
        if ((ret = nc_enddef(ncid)) != NC_NOERR) {
            return ret;
        }
    }

    // all good, write data
    if ((ret = nc_put_vara_float(ncid, *varid, start, count, data)) != NC_NOERR) {
        return ret;
    }

    return 0;
}


// XXX: we could make these functions being exported by currennt_lib


size_t readNcDimension(int ncid, const char *dimName)
{
    int ret;
    int dimid;
    size_t dimlen;

    if ((ret = nc_inq_dimid(ncid, dimName, &dimid)) || (ret = nc_inq_dimlen(ncid, dimid, &dimlen)))
        throw std::runtime_error(std::string("Cannot get dimension '") + dimName + "': " + nc_strerror(ret));

    return dimlen;
}


int readNcIntArray(int ncid, const char *arrName, size_t arrIdx)
{
    int ret;
    int varid;
    size_t start[] = {arrIdx};
    size_t count[] = {1};

    int x;
    if ((ret = nc_inq_varid(ncid, arrName, &varid)) || (ret = nc_get_vara_int(ncid, varid, start, count, &x)))
        throw std::runtime_error(std::string("Cannot read array '") + arrName + "': " + nc_strerror(ret));

    return x;
}


float* readNcFloatArray(int ncid, const char *arrName, float* ptr, size_t n)
{
    int ret;
    int varid;
    size_t start[] = {0};
    size_t count[] = {n};

    if ((ret = nc_inq_varid(ncid, arrName, &varid)) || (ret = nc_get_vara_float(ncid, varid, start, count, ptr)))
        throw std::runtime_error(std::string("Cannot read array '") + arrName + "': " + nc_strerror(ret));

    return ptr;
}


float* readNcPatternArray(int ncid, const char *arrName,
        size_t begin,
        size_t n,
        size_t patternSize, int* save_varid = 0)
{
    int ret;
    int varid;
    size_t start[] = {begin, 0};
    size_t count[] = {n, patternSize};

    float* v = new float[n * patternSize];
    if ((ret = nc_inq_varid(ncid, arrName, &varid)) || (ret = nc_get_vara_float(ncid, varid, start, count, v)))
        throw std::runtime_error(std::string("Cannot read array '") + arrName + "': " + nc_strerror(ret));

    if (save_varid != 0)
        *save_varid = varid;

    return v;
}

void SplitString(const std::string& s, std::vector<std::string>& v, const std::string& c)
{
  std::string::size_type pos1, pos2;
  pos2 = s.find(c);
  pos1 = 0;
  while(std::string::npos != pos2)
  {
    v.push_back(s.substr(pos1, pos2-pos1));

    pos1 = pos2 + c.size();
    pos2 = s.find(c, pos1);
  }
  if(pos1 != s.length())
    v.push_back(s.substr(pos1));
}



int ParseDim(const std::string& dimstr, std::vector<int>& dimvec) {
    dimvec.clear();
    std::set<int> dimset;
    std::vector<std::string> commastrvec;
    SplitString(dimstr, commastrvec, ",");
    for(const std::string &tmpstr : commastrvec) {
        std::size_t found = tmpstr.find('-');
        if (found != std::string::npos){
            std::vector<std::string> rangestrvec;
            SplitString(tmpstr, rangestrvec, "-");
            if(rangestrvec.size() == 2){
                std::stringstream ss(rangestrvec[0]);
                int begindim;
                ss >> begindim;
                std::stringstream ss2(rangestrvec[1]);
                int enddim;
                ss2 >> enddim;
                for(int i=begindim; i<enddim+1; i++){
                     dimset.insert(i);
                }
            }
        } else {

                std::stringstream ss(tmpstr);
                int dim;
                ss >> dim;
                dimset.insert(dim);
        }
    }

    for(const int &i : dimset){
        if(i >= 0) {
             dimvec.push_back(i);
        }
    }
    std::sort(dimvec.begin(),dimvec.end());

    return 0;
}

void CheckDim(std::vector<int>& dimvec ,int dim, bool iscomputeall){
    if(iscomputeall){
        dimvec.clear();

        for(int i=0; i<dim; i++){
            dimvec.push_back(i);
        }

    }else{

    std::vector<int> newdimvec;
    for(const int &i : dimvec){
        if(i < dim){
            newdimvec.push_back(i);
        }
    }

    dimvec = newdimvec;
    }
}


int getSizeFromNCFile(char* ncfile, size_t& input_size, size_t& output_size){
    int ncid;
    int ret;

    if ((ret = nc_open(ncfile, NC_NOWRITE, &ncid)) != NC_NOERR) {
        cerr << "Could not open '" << ncfile<< "': " << nc_strerror(ret) << endl;
        return 1;
    }
    cout << "Reading normdata from " << ncfile<< endl;
    try {

        input_size = readNcDimension(ncid, "inputPattSize");
        cout << "Input size: " << input_size << endl;
        output_size = readNcDimension(ncid, "targetPattSize");
        cout << "Output size: " << output_size << endl;
    }
    catch (std::runtime_error err) {
        cerr << "Could not read normdata from " << ncfile<< ": " << err.what() << endl;
        return 1;
    }

    return 0;

}

int main(int argc, char** argv)
{
    string gitcommitstr = "undefined git commit information";
#ifdef GITCOMMIT
    gitcommitstr = GITCOMMIT;
    cout << gitcommitstr << endl;
#endif

    // compute means / variances
    // OR load normdata

    if (argc <3) {
        cerr << "Usage: " << argv[0] << " -i <nc-file> [ -d inputdimstr -t outputdimstr -r reference-ncfile]" << endl;
        cerr << "-i input nc file for normalization" << endl;
        //cerr << "-n do not norm output" << endl;
        cerr << "-d select input dim for normalization, eg. 1-5,9" << endl;
        cerr << "-t select output dim for normalization, eg. 1-5,9" << endl;
        cerr << "-r load mean/std value from reference-ncfile." << endl;
        cerr << "-z normalize data use mean:stdev instead of min:(max-min)" << endl;
        cerr << "-f protect from redoing normalizing if not set." << endl << endl;
        return 1;
    }

    int opt = 0;
    char* ncfile = NULL;
    char* ncfile_load_norm = NULL;
    bool std_output = true;
    bool compute_normdata = true;
    bool computeallinputdim = true;
    bool computealloutputdim = true;
    bool overwriteMeanStd = false;
    bool useMeanStd = false;
    std::vector<int> inputDimsToBeNormed;
    std::vector<int> outputDimsToBeNormed;

    while((opt = getopt(argc, argv, "d:r:t:i:f:z")) != -1) {
        switch(opt) {
            case 'i':
                ncfile = optarg;
                break;
           // case 'n':
           //     std_output = false;
           //     break;
            case 'r':
                ncfile_load_norm = optarg;
                compute_normdata = false;
                break;
            case 'd':
                computeallinputdim = false;
                std::cout <<"----"<<std::endl;
                std::cout <<optarg<<std::endl;
                std::cout <<"----"<<std::endl;
                ParseDim(std::string(optarg), inputDimsToBeNormed);
                std::cout << "input norm dim:" << std::endl;
                for(const int &i : inputDimsToBeNormed){
                    std::cout << i <<" ";
                }
                std::cout << std::endl;
                break;
            case 't':
                computealloutputdim = false;
                std::cout <<"----"<<std::endl;
                std::cout <<optarg<<std::endl;
                std::cout <<"----"<<std::endl;
                ParseDim(std::string(optarg), outputDimsToBeNormed);
                std::cout <<"output norm dim:"<<std::endl;
                for(const int &i : outputDimsToBeNormed){
                    std::cout << i <<" ";
                }
                std::cout << std::endl;
                break;
            case 'f':
                overwriteMeanStd = true;
                std::cout << "Force doing normalize even if means and stdevs exist."
                    << std::endl;
                break;
            case 'z':
                useMeanStd = true;
                break;
            case '?':
                if (optopt == 'd')
                    fprintf (stderr, "Option -%c requires an argument.\n", optopt);
                else if (isprint (optopt))
                    fprintf (stderr, "Unknown option `-%c'.\n", optopt);
                else
                    fprintf (stderr,
                            "Unknown option character `\\x%x'.\n",
                            optopt);
                return 1;

        }

    }

    int ncid, ret, dimid;

    if(!compute_normdata && (!computeallinputdim || !computealloutputdim)){
        std::cerr << "Error: not support -d/-t with -r" <<std::endl;
        return 1;
    }

    if ((ret = nc_open(ncfile, NC_WRITE, &ncid)))
        throw std::runtime_error(std::string("Could not open '") + ncfile + "': " + nc_strerror(ret));
    else if (  (ret = nc_inq_varid(ncid, "inputMeans",   &dimid)) == NC_NOERR
            && (ret = nc_inq_varid(ncid, "inputStdevs",  &dimid)) == NC_NOERR
            /*|| (ret = nc_inq_varid(ncid, "outputMeans",  &dimid)) == NC_NOERR
            && (ret = nc_inq_varid(ncid, "outputStdevs", &dimid)) == NC_NOERR */) {
        if (overwriteMeanStd)
            std::cerr << "*** WARNING: Overwriting normalized data and params!!! ***" << std::endl;
        else {
            std::cerr << "Error: data has already normalized, preserve and quit." << std::endl;
            return 1;
        }
    }

    const size_t input_size = readNcDimension(ncid, "inputPattSize");
    cout << "Input size: " << input_size << endl;
    CheckDim(inputDimsToBeNormed, input_size, computeallinputdim);

    size_t output_size = 1;
    try {
        output_size = readNcDimension(ncid, "targetPattSize");
        cout << "Output size: " << output_size << endl;
        CheckDim(outputDimsToBeNormed,output_size,computealloutputdim);
    }
    catch (...) {
        std_output = false;
        cerr << "WARNING: targetPattSize field not found, do not standardize outputs (classification task?)" << endl;
    }

    // size check
    if(!compute_normdata){
        size_t norm_input_size = 1;
        size_t norm_output_size = 1;
        int ret = getSizeFromNCFile(ncfile_load_norm, norm_input_size, norm_output_size);
        if (ret != 0){
            return -1;
        }
        if(norm_input_size != input_size){
            cerr << "set inputPattSize: "<< input_size << " != NormFile inputPattSize " <<norm_input_size<< endl;
            return -1;
        }
        if(std_output){
            if(norm_output_size != output_size){
                cerr << "set targetPattSize: "<< output_size << " != NormFile targetPattSize " <<norm_output_size<< endl;
                return -1;
            }
        }else{
            output_size = norm_output_size;
            cerr << "set targetPattSize: "<< output_size << " with NormFile targetPattSize " <<norm_output_size<< endl;
        }
    }

    try{
        int varid;
        if (ret = nc_inq_varid(ncid, "targetPatterns", &varid)){
            throw std::runtime_error(std::string("Cannot read array 'targetPatterns': ") + nc_strerror(ret));
        }
    } catch (...) {
        std_output = false;
        cerr << "WARNING: targetPatterns field not found, skip norm output" << endl;
    }

    size_t total_sequences = readNcDimension(ncid, "numSeqs");
    cout << "# of sequences: " << total_sequences << endl;

    std::cout <<"input norm dim:"<<std::endl;
    for(const int &i : inputDimsToBeNormed){
        std::cout <<i <<" ";
    }
    std::cout <<std::endl;

    std::cout <<"output norm dim:"<<std::endl;
    for(const int &i : outputDimsToBeNormed){
        std::cout <<i <<" ";
    }
    std::cout <<std::endl;

    double input_means_tmp[input_size];
    double input_sqmeans_tmp[input_size];

    double output_means_tmp[output_size];
    double output_sqmeans_tmp[output_size];

    float input_means[input_size];
    float input_sds[input_size];

    float output_means[output_size];
    float output_sds[output_size];

    fill_n(input_means, input_size, 0.0f);
    fill_n(input_means_tmp, input_size, 0.0f);
    fill_n(output_means, output_size, 0.0f);
    fill_n(output_means_tmp, output_size, 0.0f);
    fill_n(input_sds, input_size, 1.0f);
    fill_n(input_sqmeans_tmp, input_size, 1.0f);
    fill_n(output_sds, output_size, 1.0f);
    fill_n(output_sqmeans_tmp, output_size, 1.0f);

    float input_mins[input_size];
    float input_maxes[input_size];
    float input_spans[input_size];
    float output_mins[output_size];
    float output_maxes[output_size];
    float output_spans[output_size];

    fill_n(input_mins, input_size, numeric_limits<float>::max());
    fill_n(input_maxes, input_size, numeric_limits<float>::lowest());
    fill_n(output_mins, output_size, numeric_limits<float>::max());
    fill_n(output_maxes, output_size, numeric_limits<float>::lowest());

    if (compute_normdata) {

        // compute means/variances
        size_t start = 0;
        int total_timesteps = 0;
        for (size_t i = 0; i < total_sequences; ++i) {
            size_t seqLength = readNcIntArray(ncid, "seqLengths", i);
            float* inputs = readNcPatternArray(ncid, "inputs", start, seqLength, input_size);
            //cout << "seq #" << i << " length = " << seqLength << endl;
            for (int t = 0; t < seqLength; ++t) {
                for (int j = 0; j < input_size; ++j) {
                    float tmp = inputs[t * input_size + j];
                    /* Use rapid calculation according to Welford, 1962 */
                    double k = t + total_timesteps + 1;
                    double tmp2 = tmp - input_means_tmp[j];
                    input_means_tmp[j]   += tmp2 / k;
                    input_sqmeans_tmp[j] += tmp2 * (tmp - input_means_tmp[j]); //(k - 1) / k * (tmp - input_means_tmp[j]);
                    if(tmp < input_mins[j]) input_mins[j] = tmp;
                    if(tmp > input_maxes[j]) input_maxes[j] = tmp;
                }
            }
            if (std_output) {
                float* outputs = readNcPatternArray(ncid, "targetPatterns", start, seqLength, output_size);
                //cout << "seq #" << i << " length = " << seqLength << endl;
                for (int t = 0; t < seqLength; ++t) {
                    for (int j = 0; j < output_size; ++j) {
                        float tmp = outputs[t * output_size + j];
                        //cout << "value = " << tmp << endl;
                        double k = t + total_timesteps + 1;
                        double tmp2 = tmp - output_means_tmp[j];
                        output_means_tmp[j]   += tmp2 / k;
                        output_sqmeans_tmp[j] += tmp2 * (tmp - output_means_tmp[j]);
                        if(tmp < output_mins[j]) output_mins[j] = tmp;
                        if(tmp > output_maxes[j]) output_maxes[j] = tmp;
                        //cout << "osqmeans[" << j << "] = " << output_sqmeans_tmp[j] << endl;
                    }
                }
                delete[] outputs;
            }
            delete[] inputs;
            total_timesteps += seqLength;
            start += seqLength;
        }
        float norm = 1.0 / float(total_timesteps);
        float norm2 = std::sqrt(float(total_timesteps) / float(total_timesteps - 1));
        for(size_t j = 0; j < input_size; j++) {
            if(find(inputDimsToBeNormed.begin(), inputDimsToBeNormed.end(), j) != inputDimsToBeNormed.end()){
                input_means[j] = static_cast<float>(input_means_tmp[j]);
                input_sds[j] = std::sqrt(input_sqmeans_tmp[j] / (total_timesteps - 1));
                cout << "input feature #" << j << ": mean = " << input_means[j]
                     << " +/-" << input_sds[j];
                float xmin = input_mins[j];
                float xmax = input_maxes[j];
                input_mins[j] = (99 * xmin - xmax) / 98;
                input_spans[j] = 100 * (xmax - xmin) / 98;
                if(xmin == xmax){
                    if(xmax != 0.0f){
                        input_mins[j] = 0.0f;
                        input_spans[j] = 2.0f*xmax;
                    }else{
                        input_mins[j] = -0.5f;
                        input_spans[j] = 1.0f;
                    }
                }
                cout << ": min = " << input_mins[j] << " + " << input_spans[j] << endl;
            } else {
                input_mins[j] = 0.0f;
                input_spans[j] = 1.0f;
            }
        }
        if (std_output) {
            for(const int &j : outputDimsToBeNormed){
            }
            for(size_t j = 0; j < output_size; j++) {
                if(find(outputDimsToBeNormed.begin(), outputDimsToBeNormed.end(), j) != outputDimsToBeNormed.end()){
                    output_means[j] = static_cast<float>(output_means_tmp[j]);
                    output_sds[j] = std::sqrt(output_sqmeans_tmp[j] / (total_timesteps - 1));
                    cout << "output feature #" << j << ": mean = " << output_spans[j]
                         << " +/-" << output_sds[j];
                    float xmin = output_mins[j];
                    float xmax = output_maxes[j];
                    output_mins[j] = (99 * xmin - xmax) / 98;
                    output_spans[j] = 100 * (xmax - xmin) / 98;
                    if(xmin == xmax){
                        if(xmax != 0.0f){
                            output_mins[j] = 0.0f;
                            output_spans[j] = 2.0f*xmax;
                        }else{
                            output_mins[j] = -0.5f;
                            output_spans[j] = 1.0f;
                        }
                    }
                    cout << ": min = " << output_mins[j] << " + " << output_spans[j] << endl;
                } else {
                    output_mins[j] = 0.0f;
                    output_spans[j] = 1.0f;
                }
            }
        }

    } else {
        int ncid_norm;
        if ((ret = nc_open(ncfile_load_norm, NC_NOWRITE, &ncid_norm)) != NC_NOERR) {
            cerr << "Could not open '" << ncfile_load_norm << "': " << nc_strerror(ret) << endl;
            return 1;
        }
        cout << "Reading normdata from " << ncfile_load_norm << endl;
        try {
            readNcFloatArray(ncid_norm, "inputMeans", useMeanStd ? input_means : input_mins, input_size);
            readNcFloatArray(ncid_norm, "inputStdevs", useMeanStd ? input_sds : input_spans, input_size);
            for (int j = 0; j < input_size; ++j) {
                cout << "input feature #" << j
                    << (useMeanStd ? ": mean = " : ": min = ")
                    << (useMeanStd ? input_means[j] : input_mins[j])
                    << " +/-" << (useMeanStd ? input_sds[j] : input_spans[j])
                    << endl;
            }

            // try to retrieve specified variable
            cout <<"load outputmean/std"<<endl;
            readNcFloatArray(ncid_norm, "outputMeans", useMeanStd ? output_means : output_mins, output_size);
            readNcFloatArray(ncid_norm, "outputStdevs", useMeanStd ? output_sds : output_spans, output_size);

            for (int j = 0; j < output_size; ++j) {
                cout << "output feature #" << j
                    << (useMeanStd ? ": mean = " : ": min = ")
                    << (useMeanStd ? output_means[j] : output_mins[j])
                    << " +/-" << (useMeanStd ? output_sds[j] : output_spans[j])
                    << endl;
            }
        }
        catch (std::runtime_error err) {
            cerr << "Could not read normdata from " << argv[2] << ": " << err.what() << endl;
            return 1;
        }
    }

    // save normdata into nc file given by first argument
    cout << "save normdata" << endl;
    int input_means_varid;
    ret = createVarIfNotExists(ncid, "inputMeans", "inputPattSize", &input_means_varid, useMeanStd ? input_means : input_mins);
    if (ret != NC_NOERR) {
       cerr << "ERROR saving inputMeans: " << nc_strerror(ret) << endl;
       return ret;
    }
    int input_stdevs_varid;
    ret = createVarIfNotExists(ncid, "inputStdevs", "inputPattSize", &input_stdevs_varid, useMeanStd ? input_sds : input_spans);
    if (ret != NC_NOERR) {
       cerr << "ERROR saving inputStdevs: " << nc_strerror(ret) << endl;
       return ret;
    }

    int output_dimid;
    cout <<"save outputmean/std"<<endl;
    ret = createDimIfNotExists(ncid, "targetPattSize", &output_dimid, output_size);
    if (ret != NC_NOERR) {
        cerr << "ERROR saving targetPattSize: " << nc_strerror(ret) << endl;
        return ret;
    }

    int output_means_varid;
    ret = createVarIfNotExists(ncid, "outputMeans", "targetPattSize", &output_means_varid, useMeanStd ? output_means : output_mins);
    if (ret != NC_NOERR) {
        cerr << "ERROR saving outputMeans: " << nc_strerror(ret) << endl;
        return ret;
    }
    int output_stdevs_varid;
    ret = createVarIfNotExists(ncid, "outputStdevs", "targetPattSize", &output_stdevs_varid, useMeanStd ? output_sds : output_spans);
    if (ret != NC_NOERR) {
        cerr << "ERROR saving outputStdevs: " << nc_strerror(ret) << endl;
        return ret;
    }

    // perform normalization on file pointed to by ncid
    size_t start = 0;
    for (size_t i = 0; i < total_sequences; ++i) {
        size_t seqLength = readNcIntArray(ncid, "seqLengths", i);
        int inputs_varid;
        float *inputs = readNcPatternArray(ncid, "inputs", start, seqLength, input_size, &inputs_varid);
        float *to_sub = useMeanStd ? input_means : input_mins;
        float *to_div = useMeanStd ? input_sds : input_spans;
        for (int t = 0; t < seqLength; ++t) {
            for(const int &j : inputDimsToBeNormed){
                int idx = t * input_size + j;
                inputs[idx] -= to_sub[j];
                inputs[idx] /= to_div[j];
            }
        }
        size_t starts[] = {start, 0};
        size_t counts[] = {seqLength, input_size};
        if ((ret = nc_put_vara_float(ncid, inputs_varid, starts, counts, inputs)) != NC_NOERR) {
            cerr << "Could not write standardized inputs: " << nc_strerror(ret) << endl;
            return ret;
        }
        if (std_output) {
            int outputs_varid;
            float* outputs = readNcPatternArray(ncid, "targetPatterns", start, seqLength, output_size, &outputs_varid);
            to_sub = useMeanStd ? output_means : output_mins;
            to_div = useMeanStd ? output_sds : output_spans;
            //cout << "seq #" << i << " length = " << seqLength << endl;
            for (int t = 0; t < seqLength; ++t) {
                for(const int &j : outputDimsToBeNormed){
                    int idx = t * output_size + j;
                    outputs[idx] -= to_sub[j];
                    outputs[idx] /= to_div[j];
                }
            }
            counts[1] = output_size;
            if ((ret = nc_put_vara_float(ncid, outputs_varid, starts, counts, outputs)) != NC_NOERR) {
                cerr << "Could not write standardized outputs: " << nc_strerror(ret) << endl;
                return ret;
            }
            delete[] outputs;
        }
        delete[] inputs;
        start += seqLength;
    }


    nc_close(ncid);
}
