/**
 * Swap and copy
 * 
 * @param {*} array 
 * @param {*} i 
 * @param {*} j 
 * @returns 
 */
function swappy(array, i, j) {
    const copy = [...array];

    copy[i] = array[j];
    copy[j] = array[i];

    return copy;
}

class Tensor {
    /** @type {number[]} */ _strides;
    /** @type {Float32Array} */ _data;

    /**
     * @param {number[]} shape
     * @param {Float32Array} data
     */
    constructor(shape, data = null) {
        this.shape = shape.length ? shape : (shape = [1]);

        // Initialise the 1D strides array
        this._strides = new Array(shape.length+1).fill(1);
        this._strides[shape.length-1] = shape[shape.length-1];

        for (let i = this._strides.length-3; i >= 0; i--) {
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

        // Calculate the index from _strides
        for (let i = 0; i < indices.length; i++) {
            index += this._strides[i] * indices[i];
        }

        // Retrieve the column to be indexed
        return this._data.slice(index, index + this._strides[indices.length-1]);
    }
}

class NumTensor extends Tensor {

    constructor(shape, data = null) {
        super(shape, data);

        if (!data) {
            this._data.fill(0);
        }
    }

    #reverse(index) {
        let indices = []

        for (let i = 0; i < this.shape.length; i++) {
            indices.push(Math.floor(index / this._strides[i]));
            index %= this._strides[i];
        }

        return indices;
    }

    permute(dim1, dim2) {
        const result = new NumTensor(swappy(this.shape, dim1, dim2));

        for (let i = 0; i < this._data.length; i++) {
            const indices = this.#reverse(i);

            [indices[dim1], indices[dim2]] = [indices[dim2], indices[dim1]];

            // Set at the result array
            let index = 0;

            // Calculate the index from _strides
            for (let i = 0; i < indices.length; i++) {
                index += result._strides[i] * indices[i];
            }

            result._data[index] = this._data[i];
        }

        return result;
    }

    get average() {
        return Math.floor(this._data.reduce((x, y) => x + y) / this._data.length);
    }

    add(tensor) {
        const result = new NumTensor([...this.shape]);

        for (let i = 0; i < result._data.length; i++) {
            result._data[i] = this._data[i] + tensor._data[i];
        }

        return result;
    }

    #add_fast(array1, array2) {

        for (let i = 0; i < array1.length; i++) {
            array1[i] += array2[i];
        }

        return array1;
    }

    sub(tensor) {
        const result = new NumTensor([...this.shape]);

        for (let i = 0; i < result._data.length; i++) {
            result._data[i] = this._data[i] - tensor._data[i];
        }

        return result;
    }

    s_mul(scalar) {
        let result = new NumTensor([...this.shape]);

        for (let i = 0; i < result._data.length; i++) {
            result._data[i] = this._data[i] * scalar;
        }

        return result;
    }

    #s_mul_fast(array, scalar) {
        
        for (let i = 0; i < array.length; i++) {
            array[i] *= scalar;
        }

        return array;
    }

    pow(scalar) {
        const result = new NumTensor([...this.shape]);

        for (let i = 0; i < result._data.length; i++) {
            result._data[i] = Math.pow(this._data[i], scalar);
        }

        return result;
    }

    hadamard(tensor) {
        let result = new NumTensor([this._data.length]);

        for (let i = 0; i < result._data.length; i++) {
            result._data[i] = this._data[i] * tensor._data[i];
        }

        return result;
    }

    /**
     * Computes a row-vector product on the tensor
     * 
     * @param {*} vector 
     * @returns 
     */
    v_row_mul(vector) {
        let result = new NumTensor(this.shape.slice(1));

        // slice, scale and sum each sub-tensor
        for (let i = 0; i < this.shape[0]; i++) {
            result.#add_fast(result._data, this.#s_mul_fast(this.at([i]), vector[i]));
        }

        return result;
    }

    t_mul(tensor) {
        const result = new NumTensor([...this.shape.slice(0, -1), ...tensor.shape.slice(1)]);

        const iStep = this._strides.at(-1) * this.shape.at(-1);
        const jStep = result._strides.at(-tensor.shape.length);
        
        for (let i = 0, j = 0; i < this._data.length;) {
            const vectorData = this._data.subarray(i, (i += iStep));
            const resultData = result._data.subarray(j, (j += jStep));
            this.#add_fast(resultData, tensor.v_row_mul(vectorData)._data);
        }

        return result;
    }

    outer(tensor) {
        const result = new NumTensor([...this.shape, ...tensor.shape]);
        
        for (let i = 0, j = 0; i < this._data.length; i++) {
            const resultData = result._data.subarray(j, (j += tensor._data.length));
            this.#add_fast(resultData, this.#s_mul_fast([...tensor._data], this._data[i]));
        }

        return result;
    }
}

export { Tensor, NumTensor };