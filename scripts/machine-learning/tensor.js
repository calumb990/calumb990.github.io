class Tensor {
    /** @type {number[]} */ _strides;
    /** @type {Float32Array} */ _data;

    /**
     * @param {number[]} shape
     * @param {Float32Array} data
     */
    constructor(shape, data = null) {
        this.shape = shape;

        // Initialise the 1D strides array
        this._strides = new Array(shape.length).fill(1);
        this._strides[shape.length-1] = shape[shape.length-1];

        for (let i = this._strides.length - 2; i >= 0; i--) {
            this._strides[i] = this._strides[i+1] * shape[i];
        }

        // Initialise the 1D float32 array
        const length = this._strides.shift();

        if (data && data.length !== length) {
            throw new Error("invalid input data");
        }

        this._data = data ?? new Float32Array(length);
    }

    at(indices) {
        let index = 0;

        // If number, simply index the array
        if (typeof(indices) === "number") {
            return this._data[indices];
        }

        // Calculate the index from _strides
        for (let i = 0; i < indices.length-1; i++) {
            index += this._strides[i] * indices[i];
        }

        // If only a part of the array, return 
        if (indices.length === this.shape.length) {
            return this._data[index + indices.at(-1)];
        }

        // Retrieve the column to be indexed
        return new TensorView(
            this.shape.slice(indices.length),
            this._data.subarray(index, index + this._strides.at(-1))
        );
    }
}

class TensorView {
    /** @type {Tensor} */ #tensor

    constructor(tensor) {
        this.#tensor = tensor;
    }

    at(indices) {
        this.#tensor.at(indices);
    }

    add(tensor) {
        this.#tensor.add(tensor);
    }

    s_mul(scalar) {
        this.#tensor.s_mul(scalar);
    }

    v_mul(vector) {
        this.#tensor.v_mul(vector);
    }
}


class NumTensor extends Tensor {

    constructor(tensor, requires_grad = false) {
        super(tensor, requires_grad);

        // Initialise to zeroes
        this._data.fill(0);
    }

    add(tensor) {
        const result = new Tensor(this.shape);

        for (let i = 0; i < result._data.length; i++) {
            result._data[i] = this._data[i] + tensor._data[i];
        }

        return result;
    }

    add_mut(tensor) {

        for (let i = 0; i < result._data.length; i++) {
            this._data[i] += tensor._data[i];
        }

        return this;
    }

    s_mul(scalar) {
        let result = new Tensor(this.shape);

        for (let i = 0; i < result._data.length; i++) {
            result._data[i] = this._data[i] * scalar;
        }

        return result;
    }

    s_mul_mut(scalar) {

        for (let i = 0; i < result._data.length; i++) {
            this._data[i] *= scalar;
        }

        return this;
    }

    v_mul(vector) {
        let result = new NumTensor(this.shape.slice(1));

        // slice, scale and sum each sub-tensor
        for (let i = 0; i < this.shape[0]; i++) {
            result.add_mut(this.at([i]).s_mul_mut(vector[i]));
        }

        return result;
    }

    v_mul_mut(vector) {

        // slice, scale and sum each sub-tensor
        for (let i = 0; i < this.shape[0]; i++) {
            this.add_mut(this.at([i]).s_mul_mut(vector[i]));
        }

        return this;
    }

    t_mul(tensor) {
        const result = new NumTensor(this.shape.slice(0, -1) + tensor.shape.slice(1));

        let vRow = 0;
        let rRow = 0;

        const vIdx = this._strides[this.shape.length-1];
        const rIdx = result._strides[this.shape.length-1];

        while (vRow < this._data.length) {
            const vectorData = this._data.subarray(vRow, (vRow += vIdx));
            const resultData = result._data.subarray(rRow, (rRow += rIdx));

            const one = new Tensor([vector.length], vectorData);
            const two = new Tensor(tensor.shape.slice(1), resultData);

            two.add_mut(tensor.v_mul_mut(one))
        }

        return result;
    }
}

export { Tensor, DuelTensor, NumTensor };
