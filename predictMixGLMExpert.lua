-- Prediction code for mixture of Linear regression experts
-- Shakir, May 2013

function predictMixGLMExpert(X,W,V, sigma)
	-- data must already be centred (or have preprocessing applied)

	local K = W:size(2);
	local N = X:size(1);

	-- COmpute mixing coeff: softmax(XV)
	local prod = torch.mm(X,V);
	local lse = torch.log(torch.sum(torch.exp(prod),2));
	local logSoftmax = prod - torch.repeatTensor(lse,1,K);
	local weight = torch.exp(logSoftmax)
	
	local muk = torch.Tensor(N,K):zero();
	local mu = torch.Tensor(N,1):zero();
	local sigk = torch.Tensor(N,K):zero();
	local sig = torch.Tensor(N,1):zero();
	
	-- Compute mean and var:
	-- E[x] = (r_k^T)(mu_k); Cov[x] = r_k*(sig_k + mu_k^2) - E[x]E[x]^T
	for k =1,K do
		muk[{{},{k}}] = X*W[{{},{k}}];
		mu = mu + torch.cmul(weight[{{},{k}}],muk[{{},{k}}]);
		sigk[{{},{k}}] = torch.repeatTensor(sigma[{{},{k}}],N,1);
		tmp = sigk[{{},{k}}] + torch.pow(muk[{{},{k}}],2);
		sig = sig + torch.cmul(weight[{{},{k}}],tmp);
	end;
	sig = sig - torch.pow(mu,2)
	
	return muk, sigk, mu, sig, weight;
end;

