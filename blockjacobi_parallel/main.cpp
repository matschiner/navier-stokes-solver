
/*

  Solver for the linear hyperbolic equation

  du/dt  +  div (b u) = 0

  by an explicit time-stepping method

*/



#include <solve.hpp>
#include <python_ngstd.hpp>
#


using namespace ngsolve;
using ngfem::ELEMENT_TYPE;


class BlockJacobiParallel {
public:

    const shared_ptr<SparseMatrix<double>> mat;

    BlockJacobiParallel(shared_ptr<SparseMatrix<double>> mat, py::list blocks) {
        cout << "ljslfdjl" << endl;
        
        LocalHeap lh(1000000);
        //FlatMatr
        // matrix initialize
        const auto &mat_local = *mat;

        auto comm= mat->GetParallelDofs()->GetCommunicator();
        int ranksize = comm.Size();
        int rankid = comm.Rank();
        Vector<int> shared_facets_counts(ranksize);
        shared_facets_counts = 0;
        
        FlatVector<> shared_facets[ranksize];
        for (auto block : blocks) {
            for (auto i: block){
                auto dp = mat->GetParallelDofs()->GetDistantProcs(i.cast<int>());
                for (auto proc: dp){
                    shared_facets_counts[proc]++;
                }
            }
        }
        for (auto proc: Range(ranksize)){
            shared_facets[proc]=FlatVector<>(shared_facets_counts[proc], lh);
        }
        for (auto block : blocks) {
            for (auto i: block){
                auto i_local=i.cast<int>();
                auto dp = mat->GetParallelDofs()->GetDistantProcs(i_local);
                for (auto proc: dp){
                    shared_facets[proc][i_local]=mat_local(i_local,i_local);
                }
            }
        }
        
        
        for (auto block: blocks){
            FlatMatrix tmp(py::len(block), py::len(block), lh);
            auto i_new = 0;
            auto j_new = 0;
            for (auto i: block) {
                for (auto j: block) {
                    tmp(i_new++, j_new++) = mat_local(i.cast<int>(), j.cast<int>());
                }
            }
            CalcInverse(tmp);
        }
    }


};


PYBIND11_MODULE(blockjacobi_parallel, m) {

    py::class_<BlockJacobiParallel>(m, "BlockJacobiParallel")
            .def(py::init<shared_ptr<SparseMatrix<double>>, py::list>());
//.def("Apply", &BlockJacobiParallel<2>::Apply);
}












