// #include <mpi.h>
// #include <bits/stdc++.h>

// using namespace std;

// struct Contact {
//     string name;
//     string phone;
// };

// // Reads phonebook files and returns vector of Contact structs
// vector<Contact> read_phonebook_files(int argc, char **argv) {
//     vector<Contact> contacts;
//     for (int i = 1; i < argc - 1; i++) {
//         ifstream file(argv[i]);
//         if (!file) {
//             cerr << "Error opening file: " << argv[i] << "\n";
//             continue;
//         }
//         string name, phone;
//         while (file >> name >> phone) {
//             contacts.push_back({name, phone});
//         }
//     }
//     return contacts;
// }

// // Searches for matching contact names
// string search_contacts(const vector<Contact> &contacts, const string &search_term) {
//     stringstream results;
//     for (const auto &c : contacts) {
//         if (c.name.find(search_term) != string::npos) {
//             results << c.name << " " << c.phone << "\n";
//         }
//     }
//     return results.str();
// }

// int main(int argc, char **argv) {
//     MPI_Init(&argc, &argv);
//     int rank, size;

//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &size);

//     if (argc < 3) {
//         if (rank == 0)
//             cerr << "Usage: mpirun -np <n> " << argv[0] << " <file1> ... <fileN> <search_term>\n";
//         MPI_Finalize();
//         return 1;
//     }

//     string search_term(argv[argc - 1]);
//     int total_contacts = 0;
//     vector<Contact> contacts;

//     if (rank == 0) {
//         contacts = read_phonebook_files(argc, argv);
//         total_contacts = contacts.size();
//     }

//     // Broadcast search term to all processes
//     int term_len = search_term.size();
//     MPI_Bcast(&term_len, 1, MPI_INT, 0, MPI_COMM_WORLD);
//     if (rank != 0) search_term.resize(term_len);
//     MPI_Bcast(&search_term[0], term_len, MPI_CHAR, 0, MPI_COMM_WORLD);

//     // Broadcast total number of contacts
//     MPI_Bcast(&total_contacts, 1, MPI_INT, 0, MPI_COMM_WORLD);

//     // Determine chunk for each process
//     int chunk = (total_contacts + size - 1) / size;
//     int start_idx = rank * chunk;
//     int end_idx = min(start_idx + chunk, total_contacts);
//     int local_count = end_idx - start_idx;

//     vector<Contact> local_contacts(local_count);

//     // Distribute contact data from root
//     if (rank == 0) {
//         for (int proc = 1; proc < size; proc++) {
//             int s = proc * chunk;
//             int e = min(s + chunk, total_contacts);
//             int count = e - s;
//             for (int i = s; i < e; ++i) {
//                 int len_name = contacts[i].name.size();
//                 int len_phone = contacts[i].phone.size();
//                 MPI_Send(&len_name, 1, MPI_INT, proc, 0, MPI_COMM_WORLD);
//                 MPI_Send(contacts[i].name.c_str(), len_name, MPI_CHAR, proc, 0, MPI_COMM_WORLD);
//                 MPI_Send(&len_phone, 1, MPI_INT, proc, 0, MPI_COMM_WORLD);
//                 MPI_Send(contacts[i].phone.c_str(), len_phone, MPI_CHAR, proc, 0, MPI_COMM_WORLD);
//             }
//         }
//         for (int i = start_idx; i < end_idx; ++i) {
//             local_contacts[i - start_idx] = contacts[i];
//         }
//     } else {
//         for (int i = 0; i < local_count; ++i) {
//             int len_name, len_phone;
//             MPI_Recv(&len_name, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//             string name(len_name, ' ');
//             MPI_Recv(&name[0], len_name, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//             MPI_Recv(&len_phone, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//             string phone(len_phone, ' ');
//             MPI_Recv(&phone[0], len_phone, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//             local_contacts[i] = {name, phone};
//         }
//     }

//     // 🔽 TIMING ONLY LOCAL SEARCH (correct placement)
//     MPI_Barrier(MPI_COMM_WORLD);  // ensure all reach here before timing
//     double start_time = MPI_Wtime();

//     string local_result = search_contacts(local_contacts, search_term);

//     double end_time = MPI_Wtime();

//     double local_time = end_time - start_time;

//     /*
//     MPI rule:
//     Even though all processes must call MPI_Gather, only the root process (rank 0) receives data into the recv_counts array.
//     */

//     // Send results to root
//     int local_len = local_result.size();
//     vector<int> recv_counts(size); // contains the number of elemnents each rank sends.
//     MPI_Gather(&local_len, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

//     // Compute displacements
//     // displs[] contains the starting offset in the receive buffer for each process
//     vector<int> displs(size);
//     int total_result_len = 0;
//     if (rank == 0) {
//         for (int i = 0; i < size; ++i) {
//             displs[i] = total_result_len;
//             total_result_len += recv_counts[i];
//         }
//     }

//     string global_result(total_result_len, ' ');

//     // gathers the variable-sized data to the root.
//     MPI_Gatherv(local_result.data(), local_len, MPI_CHAR,
//                 &global_result[0], recv_counts.data(), displs.data(), MPI_CHAR,
//                 0, MPI_COMM_WORLD);

//     if (rank == 0) {
//         cout << "\nSearch Results:\n" << global_result;
//     }

//     printf("Process %d execution time: %.6f seconds.\n", rank, local_time);

//     MPI_Finalize();
//     return 0;
// }


/*
    How to run:
    mpic++ -o search phonebook_search.cpp
    mpirun -np 4 ./search phonebook1.txt Bob

    time mpirun -np 8 --oversubscribe --bind-to core:overload-allowed --map-by core ./ps labtest_dataset1.txt Bob
    
*/

//Simplified
#include <bits/stdc++.h>
#include <mpi.h>
using namespace std;

struct Contact {
    string name;
    string phone;
};

void send_string(const string &text, int receiver) {
    int len = text.size() + 1;
    MPI_Send(&len, 1, MPI_INT, receiver, 1, MPI_COMM_WORLD);
    MPI_Send(text.c_str(), len, MPI_CHAR, receiver, 1, MPI_COMM_WORLD);
}

string receive_string(int sender) {
    int len;
    MPI_Recv(&len, 1, MPI_INT, sender, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    char *buf = new char[len];
    MPI_Recv(buf, len, MPI_CHAR, sender, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    string res(buf);
    delete[] buf;
    return res;
}

string vector_to_string(const vector<Contact> &contacts, int start, int end) {
    string result;
    for (int i = start; i < min((int)contacts.size(), end); i++) {
        result += contacts[i].name + "," + contacts[i].phone + "\n";
    }
    return result;
}

vector<Contact> string_to_contacts(const string &text) {
    vector<Contact> contacts;
    istringstream iss(text);
    string line;
    while (getline(iss, line)) {
        if (line.empty()) continue;
        int comma = line.find(",");
        if (comma == string::npos) continue;
        contacts.push_back({line.substr(0, comma), line.substr(comma + 1)});
    }
    return contacts;
}

string check(const Contact &c, const string &search) {
    if (c.name.find(search) != string::npos) {
        return c.name + " " + c.phone + "\n";
    }
    return "";
}

void read_phonebook(const vector<string> &files, vector<Contact> &contacts) {
    for (const string &file : files) {
        ifstream f(file);
        string line;
        while (getline(f, line)) {
            if (line.empty()) continue;
            int comma = line.find(",");
            if (comma == string::npos) continue;
            contacts.push_back({line.substr(1, comma - 2), line.substr(comma + 2, line.size() - comma - 3)});
        }
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3) {
        if (rank == 0)
            cerr << "Usage: mpirun -n <procs> " << argv[0] << " <file>... <search>\n";
        MPI_Finalize();
        return 1;
    }

    string search_term = argv[argc - 1];
    double start, end;

    if (rank == 0) {
        vector<string> files(argv + 1, argv + argc - 1);
        vector<Contact> contacts;
        read_phonebook(files, contacts);
        int total = contacts.size();
        int chunk = (total + size - 1) / size;

        for (int i = 1; i < size; i++) {
            string text = vector_to_string(contacts, i * chunk, (i + 1) * chunk);
            send_string(text, i);
        }

        start = MPI_Wtime();
        string result;
        for (int i = 0; i < min(chunk, total); i++) {
            string match = check(contacts[i], search_term);
            if (!match.empty()) result += match;
        }
        end = MPI_Wtime();

        for (int i = 1; i < size; i++) {
            string recv = receive_string(i);
            if (!recv.empty()) result += recv;
        }
        
        ofstream out("output.txt");
        out << result;
        out.close();
        printf("Process %d took %f seconds.\n", rank, end - start);

    } else {
        string recv_text = receive_string(0);
        vector<Contact> contacts = string_to_contacts(recv_text);
        start = MPI_Wtime();
        string result;
        for (auto &c : contacts) {
            string match = check(c, search_term);
            if (!match.empty()) result += match;
        }
        end = MPI_Wtime();
        send_string(result, 0);
        printf("Process %d took %f seconds.\n", rank, end - start);
    }

    MPI_Finalize();
    return 0;
}