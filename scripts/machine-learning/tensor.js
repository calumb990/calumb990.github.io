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
        this.shape = shape;

        // Initialise the 1D strides array
        this._strides = new Array(shape.length+1).fill(1);
        this._strides[shape.length-1] = shape[shape.length-1];

        for (let i = this._strides.length - 3; i >= 0; i--) {
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
        for (let i = 0; i < indices.length; i++) {
            index += this._strides[i] * indices[i];
        }

        // If only a part of the array, return 
        if (indices.length === this.shape.length) {
            return this._data[index];
        }

        // Retrieve the column to be indexed
        return this._data.slice(index, index + this._strides[indices.length-1]);
    }

    permute(dim1, dim2) {
        const result = new Tensor(swappy(this.shape, dim1, dim2));

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

    #reverse(index) {
        let indices = []

        for (let i = 0; i < this.shape.length; i++) {
            indices.push(Math.floor(index / this._strides[i]));
            index %= this._strides[i];
        }

        return indices;
    }
}

class NumTensor extends Tensor {

    constructor(shape, data = null) {
        super(shape, data);

        if (!data) {
            this._data.fill(0);
        }
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

    v_mul(vector) {
        let result = new NumTensor(this.shape.slice(1));        

        // slice, scale and sum each sub-tensor
        for (let i = 0; i < this.shape[0]; i++) {
            result.#add_fast(result._data, this.#s_mul_fast(this.at([i]), vector[i]));
        }

        return result;
    }

    t_mul(tensor) {
        const result = new NumTensor([...this.shape.slice(0, -1), ...tensor.shape.slice(1)]);

        //
        const iStep = this._strides[this.shape.length - 2];
        const jStep = result._strides[this.shape.length - 2];

        //
        for (let i = 0, j = 0; i < this._data.length;) {
            const vectorData = this._data.subarray(i, (i += iStep));
            const resultData = result._data.subarray(j, (j += jStep));
            this.#add_fast(resultData, tensor.v_mul(vectorData)._data);
        }

        return result;
    }
}

export { Tensor, NumTensor };