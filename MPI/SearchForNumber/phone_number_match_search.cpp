/*
Consider phonebook given as text files and a phone number P. Write a program using MPI to search for the person’s name who’s contact phone number is P in the phonebook. The program will generate an output file containing the line number (within input file) and person’s name with phone number P.
Input: No. of processes, phone number P
Output: Execution Time, a text file containing the line number and names with phone number P in ascending order of name
*/

#include <bits/stdc++.h>
#include <mpi.h>
using namespace std;

/*
Result structure:
একজন person-এর তথ্য রাখার জন্য
*/
struct Result {
    int line;       // file-এর line number
    string name;    // person's name
    string phone;   // phone number (space সহ)
};

/*
MPI দিয়ে string পাঠানোর function
*/
void send_string(const string &s, int dest) {
    int len = s.size() + 1;
    MPI_Send(&len, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
    MPI_Send(s.c_str(), len, MPI_CHAR, dest, 0, MPI_COMM_WORLD);
}

/*
MPI দিয়ে string receive করার function
*/
string recv_string(int src) {
    int len;
    MPI_Status status;
    MPI_Recv(&len, 1, MPI_INT, src, 0, MPI_COMM_WORLD, &status);
    char *buf = new char[len];
    MPI_Recv(buf, len, MPI_CHAR, src, 0, MPI_COMM_WORLD, &status);
    string s(buf);
    delete[] buf;
    return s;
}

int main(int argc, char **argv) {

    // ---------- 1. MPI initialize ----------
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // কোন process
    MPI_Comm_size(MPI_COMM_WORLD, &size); // total process সংখ্যা

    // ---------- 2. Input validation ----------
    // phone number space সহ দেওয়া হবে, তাই argc >= 4
    if (argc < 4) {
        if (rank == 0)
            cout << "Usage: mpirun -np <p> ./phonebook phonebook.txt 015 60 872\n";
        MPI_Finalize();
        return 0;
    }

    string filename = argv[1];

    // ---------- 3. Space-সহ phone number তৈরি ----------
    // argv[2] argv[3] argv[4] ... concatenate করা
    string search_phone = argv[2];
    for (int i = 3; i < argc; i++) {
        search_phone += " ";
        search_phone += argv[i];
    }

    vector<string> all_lines;

    // ---------- 4. Master process file পড়বে ----------
    if (rank == 0) {
        ifstream in(filename);
        string line;
        while (getline(in, line)) {
            if (!line.empty())
                all_lines.push_back(line);
        }
        in.close();
    }

    // ---------- 5. File chunk calculation ----------
    int total = all_lines.size();
    int chunk = (total + size - 1) / size;

    // ---------- 6. Master → Worker data পাঠানো ----------
    if (rank == 0) {
        for (int i = 1; i < size; i++) {
            string part;
            for (int j = i * chunk; j < min(total, (i + 1) * chunk); j++) {
                // line number সহ পাঠানো
                part += to_string(j + 1) + "|" + all_lines[j] + "\n";
            }
            send_string(part, i);
        }
    }

    // ---------- 7. Local data receive ----------
    vector<string> local;
    if (rank == 0) {
        for (int i = 0; i < min(chunk, total); i++)
            local.push_back(to_string(i + 1) + "|" + all_lines[i]);
    } else {
        string recv = recv_string(0);
        stringstream ss(recv);
        string line;
        while (getline(ss, line))
            local.push_back(line);
    }

    // ---------- 8. Search start time ----------
    double start = MPI_Wtime();

    vector<Result> found;

    // ---------- 9. Local search ----------
    for (string &l : local) {

        // line_number| "NAME","015 60 872"
        int p = l.find('|');
        int ln = stoi(l.substr(0, p));
        string rest = l.substr(p + 1);

        // quote ব্যবহার করে name এবং phone আলাদা করা
        int q1 = rest.find('"');
        int q2 = rest.find('"', q1 + 1);
        int q3 = rest.find('"', q2 + 2);
        int q4 = rest.find('"', q3 + 1);

        string name = rest.substr(q1 + 1, q2 - q1 - 1);
        string phone = rest.substr(q3 + 1, q4 - q3 - 1);

        // exact phone match
        if (phone == search_phone) {
            found.push_back({ln, name, phone});
        }
    }

    double end = MPI_Wtime();

    // ---------- 10. Worker → Master result পাঠানো ----------
    if (rank != 0) {
        string out;
        for (auto &r : found)
            out += to_string(r.line) + "," + r.name + "," + r.phone + "\n";
        send_string(out, 0);
    }

    // ---------- 11. Master result collect + sort ----------
    else {
        for (int i = 1; i < size; i++) {
            string recv = recv_string(i);
            stringstream ss(recv);
            string line;
            while (getline(ss, line)) {
                stringstream ls(line);
                string a, b, c;
                getline(ls, a, ',');
                getline(ls, b, ',');
                getline(ls, c, ',');
                found.push_back({stoi(a), b, c});
            }
        }

        // name অনুযায়ী ascending sort
        sort(found.begin(), found.end(),
             [](Result &x, Result &y) {
                 return x.name < y.name;
             });

        // ---------- 12. Output file ----------
        ofstream out("output.txt");
        for (auto &r : found)
            out << "Line " << r.line << ": "
                << r.name << " (" << r.phone << ")\n";
        out.close();

        cout << "Execution Time: " << end - start << " seconds\n";
    }

    // ---------- 13. MPI finish ----------
    MPI_Finalize();
    return 0;
}

// Execution : mpic++ phone_number_match_search.cpp -o p
// Run : mpirun -np 4 ./p phonebook.txt 011 50 173