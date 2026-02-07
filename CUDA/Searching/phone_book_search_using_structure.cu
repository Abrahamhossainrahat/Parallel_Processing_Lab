# !nvidia-smi

%%writefile search_phonebook.cu
#include<bits/stdc++.h>
#include<cuda_runtime.h>

using namespace std;

struct Contact {
  char id[50];
  char name[50];
  char number[50];
};

__device__ bool check(char* str1, char* str2, int len) {
  for(int i = 0; str1[i] != '\0'; i++) {
    int j = 0;
    while(str1[i+j] != '\0' && str2[j] != '\0' && str1[i+j] == str2[j]) {
      j++;
    }
    if(j == len-1) {
      return true;
    }
  }
  return false;
}

__global__ void searchPhonebook(Contact* phonebook, int num_contacts, char* search_name, int name_length) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < num_contacts) {
    if(check(phonebook[idx].name, search_name, name_length)) {
      printf("%s %s %s\n", phonebook[idx].id, phonebook[idx].name, phonebook[idx].number);
    }
  }
}

int main(int argc, char* argv[]) {
  if(argc != 3) {
    cerr << "Usage: " << argv[0] << "  " << endl;
    return 1;
  }

  string search_name = argv[1];
  int num_threads = atoi(argv[2]);
  // Mount Google Drive and copy the location
  string file_name = "/content/sample_data/phonebook1.txt";
  //string file_name = "phonebook1.txt";

  vector<Contact> phonebook;

  ifstream file(file_name);
  if(!file.is_open()) {
    cerr << "Error opening file: " << file_name << endl;
    return 1;
  }
  else {
    Contact contact;
    string line;
    while(getline(file, line)) {
      /* Format: "id","name","phone_number"
      int pos = line.find(",");
      strcpy(contact.id, line.substr(1, pos-2).c_str());
      line = line.substr(pos+1);
      pos = line.find(",");
      strcpy(contact.name, line.substr(1, pos-2).c_str());
      strcpy(contact.number, line.substr(pos+2, line.size()-pos-4).c_str());
      phonebook.push_back(contact);
      */

      // Format: "name","phone_number"
      int pos = line.find(",");
      // Extract name (without the quotes)
      strcpy(contact.name, line.substr(1, pos - 2).c_str());

      // Extract number (also without quotes)
      strcpy(contact.number, line.substr(pos + 2, line.size() - pos - 4).c_str());

      phonebook.push_back(contact);
    }
    file.close();
  }
  int num_contacts = phonebook.size();
  Contact* device_phonebook;
  cudaMalloc((void**)&device_phonebook, sizeof(Contact)*num_contacts);
  cudaMemcpy(device_phonebook, phonebook.data(), sizeof(Contact)*num_contacts, cudaMemcpyHostToDevice);

  int name_length = search_name.length() + 1;
  char* device_search_name;
  cudaMalloc((void**)&device_search_name, name_length);
  cudaMemcpy(device_search_name, search_name.c_str(), name_length, cudaMemcpyHostToDevice);

  for(int i = 0; i < num_contacts; i += num_threads) {
    int thread_count = min(num_contacts-i, num_threads);
    searchPhonebook<<<1, thread_count>>>(device_phonebook + i, thread_count, device_search_name, name_length);
    cudaDeviceSynchronize();
  }

  cudaFree(device_phonebook);
  cudaFree(device_search_name);

  return 0;
}

// For compile : !nvcc -arch=sm_75 search_phonebook.cu -o search_phonebook
// For run : !time ./search_phonebook JAHAN 100 > output1.txt
// for run(another thread) : !time ./search_phonebook JAHAN 2 > output1.txt
