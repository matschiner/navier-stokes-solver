
/*
 
 Solver for the linear hyperbolic equation
 
 du/dt  +  div (b u) = 0
 
 by an explicit time-stepping method
 
 */



#include <solve.hpp>
#include <python_ngstd.hpp>


using namespace ngsolve;
using ngfem::ELEMENT_TYPE;


class BlockJacobiParallel : public BaseMatrix {
public:
    
    const shared_ptr<SparseMatrix<double>> mat;
    const shared_ptr<ParallelDofs> pdofs;
    Table<int> blocks;
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
            if (block.Size()){
                continue;
            }
            FlatVector<> rhs(block.Size(), lh);
            FlatVector<> res(block.Size(), lh);
            i_old=0;
            for (auto i: block){
                rhs[i_old++]=x_local[i];
            }

            auto m=blocks_inverted[block_nr++];

            res=m*rhs;
            i_old=0;
            for (auto i: block){
                y_local[i]=res[i_old++];
            }
        }
    }
    
    BlockJacobiParallel(shared_ptr<SparseMatrix<double>> mat_in, py::list pyblocks)
    : mat(mat_in)
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
        
        auto block_count=py::len(pyblocks);
        auto block_size_max=0;
        for (auto block: pyblocks){
            if (py::len(block)>block_size_max){
                block_size_max=py::len(block);
            }
        }
        Table<int> blocks=Table<int>(block_count, block_size_max);
        for (auto block: pyblocks){
            j_new=0;
            for (auto dof: block){
                blocks[i_new][j_new++]=dof.cast<int>();
            }
            while(j_new<block_size_max){
                blocks[i_new][j_new++]=-1;
            }
            i_new++;
        }
        Array<FlatArray<double>> shared_blocks(ranksize);
        Array<FlatArray<double>> shared_blocks_get(ranksize);
    
        for (auto block : blocks) {
            if(block[0]<0){
                continue;
            }
            auto dp = mat->GetParallelDofs()->GetDistantProcs(block[0]);
            if (dp.Size() == 1) {
                //cout << dp << endl;
                proc = dp[0];
                shared_blocks_counts[proc]++;
                shared_blocks_length_overall[proc] += block_size_max*block_size_max;
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
       
   
        for (auto block : blocks) {
            if(block[0]<0){
                continue;
            }
            auto dp = mat->GetParallelDofs()->GetDistantProcs(block[0]);
            if (dp.Size() == 1) {
                proc = dp[0];
                
                i_new=0;
                for (auto i: block) {
                    j_new=0;
                    for (auto j: block) {
                        shared_blocks[proc][shared_blocks_offset[proc] + i_new * block.Size() + j_new] = mat_local(i, j);
                        j_new++;
                    }
                    i_new++;
                }
                shared_blocks_offset[proc] += block.Size()*block.Size();
            }
        }
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
        
        blocks_inverted.Assign(blocks.Size(), lh);
        shared_blocks_offset = 0;
        auto block_nr=0;
        for (auto block: blocks) {
            if(block[0]<0){
                continue;
            }
            FlatMatrix tmp(block.Size(), block.Size(), lh);
            
            auto dp = mat->GetParallelDofs()->GetDistantProcs(block[0]);
            proc = -1;
            auto n = 0;
            if (dp.Size() == 1) {
                proc = dp[0];
            }
        
            i_new=0;
            for (auto i: block) {
                j_new=0;
                for (auto j: block) {
                    tmp(i_new, j_new) = mat_local(i, j);
                    
                    if (proc != -1) {
                        auto index=shared_blocks_offset[proc] + i_new * block.Size() + j_new;
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
        
    }
    
  
};













