-- Loss function used for categorical disribution
-- Returns the loglikelihood (not negative) and its gradients.
-- Shakir, April 2013

function softmaxLoss(X,R,V)
	
	-- Dimensions
	local K = V:size(2);
	local D = X:size(2);
	grad = torch.Tensor(D,K):zero();
	
	-- 1. reweighted softmax
	local prod = torch.mm(X,V);
	local lse = torch.log(torch.sum(torch.exp(prod),2));
	local logSoftmax = prod - torch.repeatTensor(lse,1,K);
	local tt = torch.cmul(R,logSoftmax);
	val = torch.sum(tt);	
	
	-- 2. Compute the gradient
	local sm = torch.exp(logSoftmax);	

	for k = 1,K do
		tmp = R[{{},{k}}] - sm[{{},{k}}]
		zz = torch.sum(torch.cmul(X,torch.repeatTensor(tmp,1,D)),1);
	    grad[{{},{k}}] = zz:t();
	end;
		
	return val, grad;
end;
