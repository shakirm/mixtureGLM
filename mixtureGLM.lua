-- Run EM algorithm for mixtures of GLM experts
-- Shakir, April 2013

require 'genData'
require 'nn'
require 'mixGLM_util'
require 'softmaxLoss'
require 'optim'
require 'predictMixGLMExpert'

local seed = 1;
torch.manualSeed(seed);
local saveOut = 1;

-- Get training data
dataName = 'wave'; --or use 'teepee';
local fname = string.format('glmExpert_%s',dataName);
local N = 1000; -- nObs
local plot = 0; -- look at data
local x,y = genData(dataName,N,plot);

-- Standardise data and add 1's for bias
local D = x:size(2);
local M = D+1; --D+1, for bias
local mu = torch.mean(x,1);
local s = torch.std(x,1);
local X1 = torch.cdiv(x - torch.repeatTensor(mu,N,1), torch.repeatTensor(s,N,1));
local XX = torch.Tensor(N,M):fill(1);
XX[{{},{1,D}}] = X1;

-- Algorithm settings
local K = 3; -- num mixture components
local lambda = 0.01; -- reg param 
local maxIter = 100; -- max EM iters
local tol = 1e-4; -- convergence
local iter = 1; -- iteration counter

-- Initialise model params: V, W, sigma (sens to init)
V = torch.mul(torch.randn(M, K),0.1); -- gating coeffs
sigma = torch.mul(torch.rand(1,K),0.1); -- noise
W = torch.mul(torch.randn(M, K),0.1); -- regression coeffs 

-- Run EM Algorithm
local done = false;
local logPrior = nn.LogSoftMax();
local lsm = nn.LogSoftMax();
local logLik = torch.Tensor(N,K):zero();
local ll = torch.Tensor(1,maxIter):zero();
while (not done) do
	iter = iter + 1;

	--******* E-Step ----------------
	-- Compute expectation of latent variables, i.e. responsibilities
	local eta = torch.mm(XX, V);
	logPrior:forward(eta)
	
	mu = torch.mm(XX,W);
	for k = 1,K do
		m = mu[{{},{k}}]
		s = torch.squeeze(sigma[{{},{k}}]);
		logLik[{{},{k}}] = normpdfln(y:t(),m:t(),s)		
	end;
		
	logPost = logPrior.output + logLik;
	lse = torch.log(torch.sum(torch.exp(logPost),2)); -- !!must replace with a good lse function
	logResp = logPost - torch.repeatTensor(lse,1,K);	
	ll[{{1},{iter}}] = torch.sum(lse); -- logLik for convergence	
	resp = torch.exp(logResp); -- responsibility
		
	--********* M-Step ----------------
	-- Computer parameter values: W, V, sigma
	
	-- 1. Solve for W and sigma by weighted least squares (for lin reg)
	for k = 1,K do
		-- local ww2 = resp[{{},{k}}];
		local ww2 = torch.Tensor(1,N+M):fill(1);
		ww2[{{1},{1,N}}] = resp[{{},{k}}];
		local R = torch.diag(torch.squeeze(torch.sqrt(ww2)));
		
		local XX3 = torch.Tensor(N+M,M):zero();
		XX3[{{1,N},{}}] = XX;
		XX3[{{N+1,N+M},{}}] = torch.diag(torch.squeeze(torch.Tensor(M,1):fill(torch.sqrt(lambda))));		
		
		local ytmp = torch.Tensor(N+M,1):zero();
		ytmp[{{1,N},{}}] = y;
 		local a = torch.mm(R,ytmp); 				
		local b = torch.mm(R,XX3);	
		local tmp = torch.gels(a,b); -- extract soln when done.
		W[{{},{k}}] = tmp[{{1,M},{}}]:t();
		
		local sumRk = torch.sum(resp[{{},{k}}]);
		local yhat = torch.mm(XX,W[{{},{k}}]);
		local sigmaTmp = torch.sum(torch.cmul(torch.pow(y - yhat,2),resp[{{},{k}}]));
		sigma[{{1},{k}}] = sigmaTmp/sumRk; -- WILL NOT WORK FOR multivariate data - fix!!
	end;
	
	-- 2. Solve for V by multinomial regression using grad optimisation
	local lamVec = torch.Tensor(M,K):fill(lambda); -- for regularisation
	-- vvec = V:clone();
	local vvec = torch.Tensor(M,K):zero();
	vvec:resize(1,M*K);
		
	-- Anonymous function to obtain gradients
	local lossFun = function(vvec)
		local Vin = vvec:clone();
		Vin:resize(M,K);
		local val, gg = softmaxLoss(XX, resp, Vin)
		grad = gg:clone();
		grad:resize(M*K); -- must NOT be resize(1,M*K)
		val = -val + torch.sum(torch.cmul(lamVec,Vin)); -- for reg
		grad = -grad + torch.mul(torch.cmul(lamVec,Vin),2);		
		return val, grad;
	end;

	-- Optimiser options
	BFGSstate = {
		maxIter = 100,
		tolFun = 1e-4,
		tolX = 1e-4,
		learningRate = 0.3};
	   
	CGstate = {}
	
	-- Very sensitive to choice of optimiser. SGD is no good.
	vnew,fs = optim.lbfgs(lossFun,vvec, BFGSstate)
	V = vnew:resize(M,K)

	-- Check convergence
	local logLikDiff = torch.squeeze(torch.abs(ll[{{1},{iter}}] - ll[{{1},{iter-1}}]));
	done = (iter >= maxIter) or (logLikDiff <= tol);

	if iter > 2 then
		-- plot loglik on screen
		gnuplot.figure(2)
		gnuplot.plot(torch.squeeze(ll[{{1},{2,iter}}]),'+-')
		gnuplot.title('Training log likelihood');
	end;
	
	print(string.format('\r Iter %d: Loglik %4.4f \r',iter,torch.squeeze(ll[{{1},{iter}}])))
	--print('Iteration',iter, 'LogLik', torch.squeeze(ll[{{1},{iter}}]))
end;
print('Training Done!');

if 1==saveOut then
	torch.save(string.format('%s.dat',fname),{W,V,sigma});
end;

-- Predict based on experts
muk, sigk, mu, sig, weight = predictMixGLMExpert(XX,W,V, sigma); -- tr data


------- VISUALISATION ---------
max, cls = torch.max(weight,2);
yh = torch.Tensor(N,1);
for n = 1,N do
	idx = torch.squeeze(cls[{{n},{1}}]);
	yh[{{n},{1}}] = muk[{{n},{idx}}];
end;

-- Prediction
gnuplot.figure(3);
gnuplot.plot({'mean',torch.squeeze(X1),torch.squeeze(yh)},{'mode',torch.squeeze(X1),torch.squeeze(mu)},{torch.squeeze(X1),torch.squeeze(y)})
gnuplot.title('Prediction');

-- Gating function
gnuplot.figure(4);
gnuplot.plot({'Expert 1',torch.squeeze(X1), torch.squeeze(weight[{{},{1}}])},{'Expert 2', torch.squeeze(X1),torch.squeeze(weight[{{},{2}}])},{'Expert 3',torch.squeeze(X1),torch.squeeze(weight[{{},{3}}])})
gnuplot.title('Gating Function Weight');

-- plot X1 vs muk (don't know how to do this with gnuplot?!?!)