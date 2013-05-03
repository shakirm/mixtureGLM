-- Some utility function.
-- Shakir, April 2013

-- Compute PDF of a Gaussian
function normpdf(x, m, S)
	-- take input, mean m, variance s
	return torch.exp(normpdfln(x,m,S));
end;

-- Compute log of Gaussian pdf
function normpdfln(x,m,v)
	-- take input, mean m, variance s
	-- x and m must be row vector, scalar variance
	
	local D = x:size(2);
	local t1 = torch.Tensor(D):fill(1);
	t1:mul(0.5*math.log(2*math.pi*v)); -- 0.5*log(2*pi)
	local t2 = torch.pow(torch.add(x,-m),2); -- (x-m)^2
	t2:div(2*v); -- t2/(2s)
	
	val = torch.add(-t1,-t2)
	return val;
end;	
