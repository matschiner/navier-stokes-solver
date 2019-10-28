
/*
 
 Solver for the linear hyperbolic equation
 
 du/dt  +  div (b u) = 0
 
 by an explicit time-stepping method
 
 */



#include <solve.hpp>
#include <python_ngstd.hpp>


using namespace ngsolve;
using ngfem::ELEMENT_TYPE;
int pylist_get(py::handle list, int index){
    auto dof_first=0;
    auto i=0;
    for (auto dof: list){
        
        if (i>=index){
            dof_first=dof.cast<int>();
            break;
        }
        i++;
    }
    return dof_first;
}

class BlockJacobiParallel : public BaseMatrix {
public:
    
    const shared_ptr<SparseMatrix<double>> mat;
    const shared_ptr<ParallelDofs> pdofs;
    const py::list blocks;
    Array<FlatMatrix<>> blocks_inverted;
    AutoVector CreateRowVector() const override {
        return make_shared<S_ParallelBaseVectorPtr<Complex>>        (mat->Width(), pdofs->GetEntrySize(), pdofs, DISTRIBUTED);
    }
    AutoVector CreateColVector() const override {
        return make_shared<S_ParallelBaseVectorPtr<Complex>>        (mat->Width(), pdofs->GetEntrySize(), pdofs, DISTRIBUTED);
    }
    ~BlockJacobiParallel(){
        
    }
    
    int VHeight() const override
    {
        return mat->VHeight();
    }
    
    int VWidth() const override
    {
        return mat->VWidth();
    }
    void MultAdd (double s, const BaseVector & x, BaseVector & y) const override
    {
        LocalHeap lh(1000000);
        const auto & xpar = dynamic_cast_ParallelBaseVector(x);
        auto & ypar = dynamic_cast_ParallelBaseVector(y);
        
        //if(x.GetParallelStatus()==CUMULATED){
        x.Distribute();
        //}
        //if(y.GetParallelStatus()==CUMULATED){
        y.Distribute();
        //}
        
        auto block_nr=0;
        auto i_old=0;
        auto y_local=ypar.GetLocalVector()->FVDouble();
        auto x_local=xpar.GetLocalVector()->FVDouble();
        for (auto block: blocks) {
            FlatVector<> rhs(py::len(block), lh);
            FlatVector<> res(py::len(block), lh);
            i_old=0;
            for (auto i: block){
                rhs[i_old++]=x_local[i.cast<int>()];
            }

            auto m=blocks_inverted[block_nr++];
            cout << "typid"<<typeid(m).name()<<endl;
            res=m*rhs;
            
            cout <<"after mult"<<m << "sdf"<<endl;
            i_old=0;
            cout << y_local.Size()<<endl;
            for (auto i: block){
                y_local[i.cast<int>()]=res[i_old++];
            }
        }
        
    }
    
    BlockJacobiParallel(shared_ptr<SparseMatrix<double>> mat_in, py::list blocks)
    : mat(mat_in), blocks(blocks)
    {
        
        LocalHeap lh(1000000);
        //FlatMatr
        // matrix initialize
        
        
        const auto &mat_local = *mat;
        auto comm = mat->GetParallelDofs()->GetCommunicator();
        int ranksize = comm.Size();
        int rankid = comm.Rank();
        
        auto proc = 0;
        auto i_new=0, j_new=0;
        
        Vector<int> shared_blocks_counts(ranksize);
        shared_blocks_counts = 0;
        
        Vector<int> shared_blocks_length_overall(ranksize);
        shared_blocks_length_overall = 0;
        
        Vector<int> shared_blocks_offset(ranksize);
        shared_blocks_offset = 0;
        
        Array<FlatArray<double>> shared_blocks(ranksize);
        Array<FlatArray<double>> shared_blocks_get(ranksize);
        
        for (auto block : blocks) {
            auto dp = mat->GetParallelDofs()->GetDistantProcs(pylist_get(block, 0));
            if (dp.Size() == 1) {
                //cout << dp << endl;
                proc = dp[0];
                shared_blocks_counts[proc]++;
                shared_blocks_length_overall[proc] += py::len(block) * py::len(block);
            }
        }
        
        for (auto proc: Range(ranksize)) {
            if (proc!=rankid){
                if(shared_blocks_length_overall[proc]>0) {
                    shared_blocks[proc].Assign(shared_blocks_length_overall[proc], lh);
                    shared_blocks_get[proc].Assign(shared_blocks_length_overall[proc], lh);
                }
            }
        }
        comm.Barrier();
        
        for (auto block : blocks) {
            auto dp = mat->GetParallelDofs()->GetDistantProcs(pylist_get(block, 0));
            if (dp.Size() == 1) {
                proc = dp[0];
                
                auto n = py::len(block);
                //cout << "proc"<<proc<<"n"<<n<<"ljdlj"<<shared_blocks_offset[proc]<<endl;
                i_new=0;
                for (auto i: block) {
                    auto i_local = i.cast<int>();
                    j_new=0;
                    for (auto j: block) {
                        auto j_local = j.cast<int>();
                        //cout << "i,j"<<i_new<<" "<<j_new<<" sdf "<<shared_blocks_offset[proc] + i_new * n + j_new<<" / "<<shared_blocks_length_overall[proc]<<endl;
                        shared_blocks[proc][shared_blocks_offset[proc] + i_new * n + j_new] = mat_local(i_local, j_local);
                        j_new++;
                    }
                    i_new++;
                }
                shared_blocks_offset[proc] += n * n;
            }
        }
        //cout <<"sljdflj"<< shared_blocks<<endl;
        Array<MPI_Request> requests;
        for (auto i : Range(ranksize)) {
            if (i == rankid || shared_blocks_counts[i]==0) {
                continue;
            }
            comm.ISend(shared_blocks[i], i, 0);
            auto r_recv = comm.IRecv(shared_blocks_get[i], i, 0);
            requests.Append(r_recv);
        }
        MyMPI_WaitAll(requests);
        
        blocks_inverted.Assign(py::len(blocks), lh);
        shared_blocks_offset = 0;
        auto block_nr=0;
        for (auto block: blocks) {
            FlatMatrix tmp(py::len(block), py::len(block), lh);
        
            auto dp = mat->GetParallelDofs()->GetDistantProcs(pylist_get(block, 0));
            proc = -1;
            auto n = 0;
            if (dp.Size() == 1) {
                proc = dp[0];
                n = py::len(block);
            }
        
            i_new=0;
            for (auto i: block) {
                j_new=0;
                for (auto j: block) {
                    tmp(i_new, j_new) = mat_local(i.cast<int>(), j.cast<int>());
                    
                    if (proc != -1) {
                        auto index=shared_blocks_offset[proc] + i_new * n + j_new;
                        tmp(i_new, j_new) += shared_blocks_get[proc][index];
                    }
                    j_new++;
                }
                i_new++;
            }
            if (dp.Size() == 1) {
                shared_blocks_offset[proc] += n * n;
            }
            
            
            CalcInverse(tmp);
            blocks_inverted[block_nr].Assign(tmp);
        }
        comm.Barrier();
    }
    
    
};


PYBIND11_MODULE(blockjacobi_parallel, m) {
    
    py::class_<BlockJacobiParallel, shared_ptr<BlockJacobiParallel>, BaseMatrix>(m, "BlockJacobiParallel")
    .def(py::init<shared_ptr<SparseMatrix<double>>, py::list>());
    //.def("Apply", &BlockJacobiParallel<2>::Apply);
}












